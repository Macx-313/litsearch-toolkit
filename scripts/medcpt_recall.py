import os
#!/usr/bin/env python3
# MedCPT 对照召回：全子集向量建临时 collection → 22 条英文查询召回 → 对比 8B
import json, time
import numpy as np
import torch
torch.set_num_threads(32)
from qdrant_client import QdrantClient, models

LT = os.environ.get("LITSEARCH_HOME", os.path.expanduser("~/litsearch"))
QD = QdrantClient(url="http://localhost:6333", timeout=300)
COL = "medcpt_v1"

vecs = np.load(f"{LT}/goldset/medcpt_full.npy", mmap_mode="r")
vpmids = [l.strip() for l in open(f"{LT}/goldset/medcpt_full.pmids.txt")]
qv = np.load(f"{LT}/goldset/medcpt_qvecs.npy")
qpmids = [l.strip() for l in open(f"{LT}/goldset/medcpt_qvecs.pmids.txt")]
print(f"medcpt corpus={len(vpmids)} dim={vecs.shape[1]}", flush=True)

try:
    QD.delete_collection(COL)
except Exception:
    pass
QD.create_collection(COL, vectors_config=models.VectorParams(
    size=vecs.shape[1], distance=models.Distance.DOT,
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, quantile=0.99, always_ram=True))))
B = 2048
for i in range(0, len(vpmids), B):
    j = min(i + B, len(vpmids))
    QD.upload_points(COL, points=[
        models.PointStruct(id=k, vector=vecs[k].tolist(), payload={"p": p})
        for k, p in enumerate(vpmids[i:j], start=i)], wait=True)
    if i % (B * 25) == 0:
        print(f"  upload {j}/{len(vpmids)}", flush=True)
print(f"uploaded={QD.get_collection(COL).points_count}", flush=True)

pmid2id = {p: i for i, p in enumerate(vpmids)}
ranks = []
for qi, tgt in enumerate(qpmids):
    r = QD.query_points(COL, query=qv[qi].tolist(), limit=100).points
    ids = [p.id for p in r]
    t = pmid2id.get(tgt)
    ranks.append(ids.index(t) + 1 if t in ids else 999)
ranks = np.array(ranks)
res = {"arm": "medcpt_768d_en", "n": len(qpmids),
       "recall@5": float((ranks <= 5).mean()), "recall@10": float((ranks <= 10).mean()),
       "recall@20": float((ranks <= 20).mean()), "recall@100": float((ranks <= 100).mean()),
       "median_rank": float(np.median(ranks)) if (ranks < 999).any() else None,
       "misses": int((ranks == 999).sum()),
       "rank_details": {p: int(r) for p, r in zip(qpmids, ranks)}}
json.dump(res, open(f"{LT}/logs/n9_medcpt_recall.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "rank_details"}, ensure_ascii=False))
QD.delete_collection(COL)
print("MEDCPT_RECALL_DONE", flush=True)
