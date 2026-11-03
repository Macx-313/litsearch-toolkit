import os
#!/usr/bin/env python3
# rerank_v2：C-2（hybrid 池 + bge，生产配置首次有数）+ N-6（qwen3r 预算对齐 2000 字符，同池同预算受控对照）
# 判定规则 v2（2026-09-27 14:48 修订，合规事前修订：修订时 logs/rerank_v2_hybrid_pool.json 不存在、双臂均未出数。
# 依据 WB 验收回执 M-1：规则 v1 的单点比值阈值（qwen3r@1 >= 95%*bge@1）在 n=22 统计上不可判定
# （Wilson 半宽 ±0.19-0.20 > 阈值本身），改为 McNemar 精确配对检验：
#   p<0.05 且 qwen3r 方向更优 ⇒ qwen3r 入候选扩测；否则「证据不足：保留 bge 为默认精排器（未决，需扩样）」
# 每臂结束即落盘（N-5）
import json, os, time
import numpy as np
import torch
torch.set_num_threads(32)
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

LT = os.environ.get("LITSEARCH_HOME", os.path.expanduser("~/litsearch"))
QD = QdrantClient(url="http://localhost:6333", timeout=300)
COL = "pubmed_v2"
OUT = f"{LT}/logs/rerank_v2_hybrid_pool.json"

texts = {}
for line in open(f"{LT}/parsed/p1_neurocrit.jsonl"):
    d = json.loads(line)
    p = str(d["pmid"])
    t = (d.get("title") or "").strip(); a = (d.get("abstract") or "").strip()
    texts[p] = (t + "\n" + a).strip() if a else t

sm = SparseTextEmbedding(model_name="Qdrant/bm25")
qv = np.load(f"{LT}/goldset/gold_queries_en_enc.npy")
qpmids = [l.strip() for l in open(f"{LT}/goldset/gold_queries_en_enc.pmids.txt")]
qtexts = {r[0]: r[1] for r in (l.rstrip("\n").split("\t") for l in open(f"{LT}/goldset/gold_queries_en.tsv"))}

def hybrid_top100(qi):
    dq = (qv[qi] / np.linalg.norm(qv[qi])).tolist()
    sq = next(sm.embed([qtexts[qpmids[qi]]]))
    sqv = models.SparseVector(indices=[int(x) for x in sq.indices], values=[float(x) for x in sq.values])
    r = QD.query_points(COL,
        prefetch=[models.Prefetch(query=dq, using="dense", limit=100),
                  models.Prefetch(query=sqv, using="sparse", limit=100)],
        query=models.FusionQuery(fusion=models.Fusion.RRF), limit=100).points
    return [p.payload["p"] for p in r]

cands = [hybrid_top100(i) for i in range(len(qpmids))]
print(f"[pool] hybrid top-100 x{len(qpmids)} ready", flush=True)

def dump(arm, ranks):
    ranks_a = np.array(ranks)
    row = {"arm": arm, "pool": "hybrid_RRF", "budget": "2000 chars", "n": len(qpmids),
           "recall@1": float((ranks_a <= 1).mean()), "recall@5": float((ranks_a <= 5).mean()),
           "recall@10": float((ranks_a <= 10).mean()),
           "median_rank": float(np.median(ranks_a)) if (ranks_a < 999).any() else None,
           "misses": int((ranks_a == 999).sum()),
           "rank_details": {p: int(r) for p, r in zip(qpmids, ranks)}}
    d = json.load(open(OUT)) if os.path.exists(OUT) else {}
    d[arm] = row
    json.dump(d, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"[{arm}] {json.dumps({k: v for k, v in row.items() if k != 'rank_details'})}", flush=True)

# 臂 1：bge（hybrid 池，en，2000 字符）
from sentence_transformers import CrossEncoder
ce = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cpu")
t0 = time.time(); bge_ranks = []
for qi, tgt in enumerate(qpmids):
    pm = [p for p in cands[qi] if p in texts]
    pairs = [(qtexts[tgt], texts[p][:2000]) for p in pm]
    scores = ce.predict(pairs, batch_size=16)
    order = np.argsort(-np.asarray(scores))
    bge_ranks.append(list(np.array(pm)[order]).index(tgt) + 1 if tgt in pm else 999)
dump("bge_hybridpool_en", bge_ranks)
print(f"bge elapsed {time.time()-t0:.0f}s", flush=True)
del ce

# 臂 2：qwen3r（同池同预算；doc 2000 字符 + max_length=1024 保证 prompt 尾部完整）
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side="left")
lm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", dtype=torch.float32)
lm.eval()
yes_id = tok("yes", add_special_tokens=False)["input_ids"][0]
no_id = tok("no", add_special_tokens=False)["input_ids"][0]
def qscore(queries, docs):
    prompts = [f"<|im_start|>system\nJudge whether the Document meets the requirements based on the Query. Only answer Yes or No.<|im_end|>\n<|im_start|>user\nQuery: {q}\nDocument: {d}<|im_end|>\n<|im_start|>assistant\n" for q, d in zip(queries, docs)]
    enc = tok(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
    with torch.no_grad():
        logits = lm(**enc).logits[:, -1, :]
    return (logits[:, yes_id] - logits[:, no_id]).tolist()
t0 = time.time(); q_ranks = []
for qi, tgt in enumerate(qpmids):
    pm = [p for p in cands[qi] if p in texts]
    docs = [texts[p][:2000] for p in pm]
    scores = qscore([qtexts[tgt]] * len(pm), docs)
    order = np.argsort(-np.asarray(scores))
    q_ranks.append(list(np.array(pm)[order]).index(tgt) + 1 if tgt in pm else 999)
dump("qwen3r_aligned_hybridpool_en", q_ranks)
print(f"qwen3r elapsed {time.time()-t0:.0f}s", flush=True)

# 判定（规则 v2：McNemar 精确配对检验，@1 与 @10 各算一份）
from math import comb
def mcnemar(a_hits, c_hits):
    b = sum(1 for a, c in zip(a_hits, c_hits) if a and not c)
    c2 = sum(1 for a, c in zip(a_hits, c_hits) if c and not a)
    nn = b + c2
    p = 1.0 if nn == 0 else min(1.0, 2 * sum(comb(nn, i) for i in range(min(b, c2) + 1)) / 2 ** nn)
    return b, c2, p
b1, c1_, p1_ = mcnemar([r <= 1 for r in bge_ranks], [r <= 1 for r in q_ranks])
b10, c10_, p10_ = mcnemar([r <= 10 for r in bge_ranks], [r <= 10 for r in q_ranks])
verdict = (f"qwen3r 入候选扩测（McNemar@1 p={p1_:.3f}，qwen3r 更优 {c1_}>{b1}）"
           if (p1_ < 0.05 and c1_ > b1) else
           f"证据不足：保留 bge 为生产精排器（未决，需扩样；McNemar@1 p={p1_:.3f} b={b1} c={c1_}；@10 p={p10_:.3f} b={b10} c={c10_}）")
d = json.load(open(OUT))
d["preregistered_verdict"] = verdict
d["preregistration_log"] = {
    "rule_v1": "qwen3r@1 >= 95%*bge@1（作废：n=22 不可判定，CI 半宽大于阈值）",
    "rule_v2": "McNemar 精确检验 p<0.05 且 qwen3r 方向更优 ⇒ 入候选；否则证据不足保留 bge（未决）",
    "amended_at": "2026-09-27 14:48", "amended_while": "双臂出数前，OUT 不存在（合规事前修订）",
    "authority": "WB 验收回执 2026-09-27-1445 M-1"}
d["_supersedes"] = "本件 qwen3r_aligned_hybridpool_en 取代 n9_rerank.json 中 INVALID_SUPERSEDED 的 qwen3r_* 两臂"
json.dump(d, open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"[VERDICT] {verdict}", flush=True)
