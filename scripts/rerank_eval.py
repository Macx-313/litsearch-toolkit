import os
#!/usr/bin/env python3
# Reranker 臂：dense top-100 候选 × 两款 reranker（bge-reranker-v2-m3 CrossEncoder / Qwen3-Reranker-0.6B causal）
# 输入查询：中文版 + 英文版各测一遍；输出目标文献精排后排名对比
import json, sys, time
import numpy as np
import torch
from qdrant_client import QdrantClient, models

torch.set_num_threads(48)
LT = os.environ.get("LITSEARCH_HOME", os.path.expanduser("~/litsearch"))
QD = QdrantClient(url="http://localhost:6333", timeout=300)
COL = "pubmed_v2"  # dense 腿用 v2 的 dense named vector

def top100(dq):
    r = QD.query_points(COL, prefetch=[models.Prefetch(query=dq, using="dense", limit=100)],
                        query=models.FusionQuery(fusion=models.Fusion.RRF), limit=100).points
    return [(p.payload["p"], p.id) for p in r]

def main(only=None):
    # 子集文本
    texts = {}
    for line in open(f"{LT}/parsed/p1_neurocrit.jsonl"):
        d = json.loads(line)
        p = str(d["pmid"])
        t = (d.get("title") or "").strip(); a = (d.get("abstract") or "").strip()
        txt = (t + "\n" + a).strip() if a else t
        texts[p] = txt[:2000]  # reranker 截断

    arms = {}
    for tag, qprefix, qtsv in [("zh", "gold_queries_enc", "gold_queries.tsv"),
                                ("en", "gold_queries_en_enc", "gold_queries_en.tsv")]:
        qv = np.load(f"{LT}/goldset/{qprefix}.npy")
        qpmids = [l.strip() for l in open(f"{LT}/goldset/{qprefix}.pmids.txt")]
        qtexts = {r[0]: r[1] for r in (l.rstrip("\n").split("\t") for l in open(f"{LT}/goldset/{qtsv}"))}
        cands = []
        for qi, tgt in enumerate(qpmids):
            dq = (qv[qi] / np.linalg.norm(qv[qi])).tolist()
            cands.append(top100(dq))
        arms[tag] = (qpmids, qtexts, cands)
        print(f"[cands] {tag} ready", flush=True)

    results = {}

    # --- bge-reranker-v2-m3（CrossEncoder）---
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cpu")
    for tag in ["zh", "en"]:
        qpmids, qtexts, cands = arms[tag]
        t0 = time.time(); ranks = []
        for qi, tgt in enumerate(qpmids):
            pairs = [(qtexts[tgt], texts[c[0]]) for c in cands[qi] if c[0] in texts]
            pm = [c[0] for c in cands[qi] if c[0] in texts]
            scores = ce.predict(pairs, batch_size=16)
            order = np.argsort(-np.asarray(scores))
            rank = list(np.array(pm)[order]).index(tgt) + 1 if tgt in pm else 999
            ranks.append(rank)
        ranks = np.array(ranks)
        results[f"bge_{tag}"] = summary(ranks, qpmids)
        print(f"bge {tag}: {time.time()-t0:.0f}s {results[f'bge_{tag}']}", flush=True)
    del ce

    # --- Qwen3-Reranker-0.6B（causal LM yes/no 打分）---
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok2 = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side="left")
    lm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float32)
    lm.eval()
    yes_id = tok2("yes", add_special_tokens=False)["input_ids"][0]
    no_id = tok2("no", add_special_tokens=False)["input_ids"][0]
    def qwen_score(queries, docs):
        prompts = [f"<|im_start|>system\nJudge whether the Document meets the requirements based on the Query. Only answer Yes or No.<|im_end|>\n<|im_start|>user\nQuery: {q}\nDocument: {d}<|im_end|>\n<|im_start|>assistant\n" for q, d in zip(queries, docs)]
        enc = tok2(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = lm(**enc).logits[:, -1, :]
        yesl = logits[:, yes_id]; nol = logits[:, no_id]
        return (yesl - nol).tolist()
    for tag in ["zh", "en"]:
        qpmids, qtexts, cands = arms[tag]
        t0 = time.time(); ranks = []
        for qi, tgt in enumerate(qpmids):
            pm = [c[0] for c in cands[qi] if c[0] in texts]
            docs = [texts[p][:600] for p in pm]
            scores = qwen_score([qtexts[tgt]] * len(pm), docs)
            order = np.argsort(-np.asarray(scores))
            rank = list(np.array(pm)[order]).index(tgt) + 1 if tgt in pm else 999
            ranks.append(rank)
        ranks = np.array(ranks)
        results[f"qwen3r_{tag}"] = summary(ranks, qpmids)
        print(f"qwen3r {tag}: {time.time()-t0:.0f}s {results[f'qwen3r_{tag}']}", flush=True)

    json.dump(results, open(f"{LT}/logs/n9_rerank.json", "w"), ensure_ascii=False, indent=1)
    print("saved -> logs/n9_rerank.json")

def summary(ranks, qpmids):
    return {"n": len(qpmids), "recall@1": float((ranks <= 1).mean()), "recall@5": float((ranks <= 5).mean()),
            "recall@10": float((ranks <= 10).mean()),
            "median_rank": float(np.median(ranks)) if (ranks < 999).any() else None,
            "misses": int((ranks == 999).sum()),
            "rank_details": {p: int(r) for p, r in zip(qpmids, ranks)}}

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else None)
