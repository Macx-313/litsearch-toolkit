import os
#!/usr/bin/env python3
# Qwen3-Reranker 重跑（截断修复版：文档截头 600 字符，保 prompt 结构完整）
import json, time
import numpy as np
import torch
torch.set_num_threads(48)
from transformers import AutoModelForCausalLM, AutoTokenizer
from qdrant_client import QdrantClient, models

LT = os.environ.get("LITSEARCH_HOME", os.path.expanduser("~/litsearch"))
QD = QdrantClient(url="http://localhost:6333", timeout=300)
COL = "pubmed_v2"

texts = {}
for line in open(f"{LT}/parsed/p1_neurocrit.jsonl"):
    d = json.loads(line)
    p = str(d["pmid"])
    t = (d.get("title") or "").strip(); a = (d.get("abstract") or "").strip()
    texts[p] = (t + "\n" + a).strip() if a else t

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side="left")
lm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", dtype=torch.float32)
lm.eval()
yes_id = tok("yes", add_special_tokens=False)["input_ids"][0]
no_id = tok("no", add_special_tokens=False)["input_ids"][0]

def score(queries, docs):
    prompts = [f"<|im_start|>system\nJudge whether the Document meets the requirements based on the Query. Only answer Yes or No.<|im_end|>\n<|im_start|>user\nQuery: {q}\nDocument: {d}<|im_end|>\n<|im_start|>assistant\n" for q, d in zip(queries, docs)]
    enc = tok(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = lm(**enc).logits[:, -1, :]
    return (logits[:, yes_id] - logits[:, no_id]).tolist()

results = {}
for tag, qprefix, qtsv in [("zh", "gold_queries_enc", "gold_queries.tsv"),
                           ("en", "gold_queries_en_enc", "gold_queries_en.tsv")]:
    qv = np.load(f"{LT}/goldset/{qprefix}.npy")
    qpmids = [l.strip() for l in open(f"{LT}/goldset/{qprefix}.pmids.txt")]
    qtexts = {r[0]: r[1] for r in (l.rstrip("\n").split("\t") for l in open(f"{LT}/goldset/{qtsv}"))}
    t0 = time.time(); ranks = []
    for qi, tgt in enumerate(qpmids):
        dq = (qv[qi] / np.linalg.norm(qv[qi])).tolist()
        r = QD.query_points(COL, prefetch=[models.Prefetch(query=dq, using="dense", limit=100)],
                            query=models.FusionQuery(fusion=models.Fusion.RRF), limit=100).points
        pm = [p.payload["p"] for p in r if p.payload["p"] in texts]
        docs = [texts[p][:600] for p in pm]
        scores = score([qtexts[tgt]] * len(pm), docs)
        order = np.argsort(-np.asarray(scores))
        rank = list(np.array(pm)[order]).index(tgt) + 1 if tgt in pm else 999
        ranks.append(rank)
    ranks = np.array(ranks)
    results[f"qwen3r_fix_{tag}"] = {"n": len(qpmids), "recall@1": float((ranks <= 1).mean()),
        "recall@5": float((ranks <= 5).mean()), "recall@10": float((ranks <= 10).mean()),
        "median_rank": float(np.median(ranks)) if (ranks < 999).any() else None,
        "misses": int((ranks == 999).sum()),
        "rank_details": {p: int(r) for p, r in zip(qpmids, ranks)}}
    print(f"qwen3r_fix {tag}: {time.time()-t0:.0f}s {json.dumps({k:v for k,v in results[f'qwen3r_fix_{tag}'].items() if k!='rank_details'})}", flush=True)

json.dump(results, open(f"{LT}/logs/n9_rerank_qwen3r_fix.json", "w"), ensure_ascii=False, indent=1)
print("saved -> logs/n9_rerank_qwen3r_fix.json", flush=True)
