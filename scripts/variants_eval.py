import os
#!/usr/bin/env python3
# 扩样评估：63 条查询（21 原版 + 42 变体）× 四臂（dense_zh-类/dense_en/hybrid/bge）
# 读取变体编码 npy（GPU 迷你窗产出），对 pubmed_v1/v2 出各臂 n=63 的 recall + 与 n=21 对照
import json, sys, statistics
import numpy as np
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder
import torch
torch.set_num_threads(48)

LT = os.environ.get("LITSEARCH_HOME", os.path.expanduser("~/litsearch"))
QD = QdrantClient(url="http://localhost:6333", timeout=120)
VOID = "23000183"

# 63 条查询清单（原版21 + 变体42，格式 pmid\tquery\tlang）
rows = [l.rstrip("\n").split("\t") for l in open(f"{LT}/goldset/gold_queries_v63.tsv") if l.strip()]
qv = np.load(f"{LT}/goldset/gold_queries_v63_enc.npy")
vpmids = [l.strip() for l in open(f"{LT}/encoded/p1_neurocrit.pmids.txt")]
pmid2id = {p: i for i, p in enumerate(vpmids)}
texts = {}
for line in open(f"{LT}/parsed/p1_neurocrit.jsonl"):
    d = json.loads(line)
    p = str(d["pmid"])
    t = (d.get("title") or "").strip(); a = (d.get("abstract") or "").strip()
    texts[p] = (t + "\n" + a).strip() if a else t
queries = [(r[0], r[1], r[2]) for r in rows]

sm = SparseTextEmbedding(model_name="Qdrant/bm25")
def en_of(pmid):
    for r in rows:
        if r[0] == pmid and r[2] == "en":
            return r[1]
    return None

def ranks_dense(i):
    r = QD.query_points("pubmed_v1", query=(qv[i] / np.linalg.norm(qv[i])).tolist(), limit=100).points
    ids = [p.id for p in r]
    return ids
def ranks_hybrid(i, pmid):
    dq = (qv[i] / np.linalg.norm(qv[i])).tolist()
    e = en_of(pmid)
    sq = next(sm.embed([e])) if e else None
    prefetch = [models.Prefetch(query=dq, using="dense", limit=100)]
    if sq:
        prefetch.append(models.Prefetch(query=models.SparseVector(indices=[int(x) for x in sq.indices], values=[float(x) for x in sq.values]), using="sparse", limit=100))
    r = QD.query_points("pubmed_v2", prefetch=prefetch, query=models.FusionQuery(fusion=models.Fusion.RRF), limit=100).points
    return [p.id for p in r]

res = {"dense": [], "hybrid": []}
cands_by_q = []
for i, (pmid, q, lang) in enumerate(queries):
    t = pmid2id.get(pmid)
    d_ids = ranks_dense(i)
    res["dense"].append(d_ids.index(t) + 1 if t in d_ids else 999)
    h_ids = ranks_hybrid(i, pmid)
    res["hybrid"].append(h_ids.index(t) + 1 if t in h_ids else 999)
    cands_by_q.append((pmid, q, h_ids))

# bge 精排臂（对 hybrid 候选）
ce = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cpu")
res["bge"] = []
for pmid, q, h_ids in cands_by_q:
    pm = [vpmids[k] if False else None for k in []]  # ids are positional ids
    pm = [vpmids[k] for k in h_ids if k < len(vpmids)]
    pm = [p for p in pm if p in texts]
    if pmid not in pm:
        res["bge"].append(999); continue
    pairs = [(q, texts[p][:2000]) for p in pm]
    scores = ce.predict(pairs, batch_size=16)
    order = np.argsort(-np.asarray(scores))
    res["bge"].append(list(np.array(pm)[order]).index(pmid) + 1)

def summ(ranks):
    r = np.array(ranks)
    return {"n": len(r), "@1": round(float((r <= 1).mean()), 3), "@5": round(float((r <= 5).mean()), 3),
            "@10": round(float((r <= 10).mean()), 3), "@100": round(float((r <= 100).mean()), 3),
            "median": float(np.median([x for x in r if x < 999])) if (r < 999).any() else None,
            "miss": int((r == 999).sum())}

out = {k: summ(v) for k, v in res.items()}
# 原版/变体分层
n0 = 21
out_layered = {}
for name, arr in res.items():
    out_layered[name] = {"orig_n21": summ(arr[:n0]), "variants_n42": summ(arr[n0:])}
out["_layered"] = out_layered
out["_design"] = "63 = 21 原版（含 v63 文件内 21 条原查询重跑）+ 42 变体（21 中文口语化 + 21 英文技术化）；bge 用混合池候选；扩样 v1（单目标 qrels）"
json.dump(out, open(f"{LT}/logs/n9_expanded_v63.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
