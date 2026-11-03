import os
#!/usr/bin/env python3
# H-4：dense+英文对照臂（闭合 MedCPT 对照的双重自变量：同查询语言、同检索层）
# 用现成 gold_queries_en_enc.npy 直查 pubmed_v1（dense-only collection），零编码零 GPU
import json, statistics
import numpy as np
from qdrant_client import QdrantClient

LT = os.environ.get("LITSEARCH_HOME", os.path.expanduser("~/litsearch"))
QD = QdrantClient(url="http://localhost:6333", timeout=60)
VOID = "23000183"

qv = np.load(f"{LT}/goldset/gold_queries_en_enc.npy")
qpmids = [l.strip() for l in open(f"{LT}/goldset/gold_queries_en_enc.pmids.txt")]
vpmids = [l.strip() for l in open(f"{LT}/encoded/p1_neurocrit.pmids.txt")]
pmid2id = {p: i for i, p in enumerate(vpmids)}
ranks = []
for qi, tgt in enumerate(qpmids):
    r = QD.query_points("pubmed_v1", query=(qv[qi] / np.linalg.norm(qv[qi])).tolist(), limit=100).points
    ids = [p.id for p in r]
    t = pmid2id.get(tgt)
    ranks.append(ids.index(t) + 1 if t in ids else 999)
rd = {p: int(r) for p, r in zip(qpmids, ranks) if p != VOID}
n = len(rd)
import math
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return round(max(0, c-h), 2), round(min(1, c+h), 2)
row = {"arm": "dense_en (Qwen3-8B, 英文直查)", "pool": "dense_top100", "n": n}
for k in (1, 5, 10, 20, 100):
    hit = sum(1 for r in rd.values() if r <= k)
    lo, hi = wilson(hit, n)
    row[f"@{k}"] = f"{hit}/{n}={hit/n:.3f} [{lo},{hi}]"
finite = [r for r in rd.values() if r < 999]
row["median(finite)"] = statistics.median(finite) if finite else None
row["miss"] = sum(1 for r in rd.values() if r == 999)
row["rank_details"] = rd
print(json.dumps({k: v for k, v in row.items() if k != "rank_details"}, ensure_ascii=False))
d = json.load(open(f"{LT}/logs/n9_final_table_v2.json"))
d["arms"]["dense_en (Qwen3-8B, 英文直查)"] = row
d["_footnotes"].append("⑦ H-4 闭合：dense_en 与 medcpt_768d 同为英文+纯 dense，唯一变量=模型——对照口径成立")
json.dump(d, open(f"{LT}/logs/n9_final_table_v2.json", "w"), ensure_ascii=False, indent=1)
print("appended -> n9_final_table_v2.json")
