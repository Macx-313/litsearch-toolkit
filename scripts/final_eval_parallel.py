import os
#!/usr/bin/env python3
# 终评并行版：ThreadPoolExecutor 12 路（93 查询独立，优化期 32s/条 → 总耗时 ~5 分钟）
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from qdrant_client import QdrantClient

LT = os.environ.get("LITSEARCH_HOME", os.path.expanduser("~/litsearch"))
QD = QdrantClient(url="http://localhost:6333", prefer_grpc=True, timeout=300)
COL = "pubmed_full"
VOID_MISSING = 888

rows = [l.rstrip("\n").split("\t") for l in open(f"{LT}/goldset/gold_queries_v93.tsv") if l.strip()]
targets = {r[0] for r in rows}
pmid2id = {}
with open(f"{LT}/parsed/full_corpus.jsonl") as f:
    for lineno, line in enumerate(f):
        p = line.split('"pmid": ', 1)[1].split(",", 1)[0]
        if p in targets:
            pmid2id[p] = lineno
print(f"targets mapped: {len(pmid2id)}/{len(targets)}", flush=True)

qv = np.load(f"{LT}/goldset/gold_queries_v93_enc.npy")
assert qv.shape[0] == len(rows)

def one(i):
    pmid = rows[i][0]
    t = pmid2id.get(pmid)
    for attempt in range(3):
        try:
            r = QD.query_points(COL, query=(qv[i] / np.linalg.norm(qv[i])).tolist(), limit=100).points
            ids = [p.id for p in r]
            return i, (ids.index(t) + 1 if t in ids else (999 if t is not None else VOID_MISSING))
        except Exception:
            if attempt == 2:
                return i, 997
import time; t0 = time.time()
ranks = [None] * len(rows)
with ThreadPoolExecutor(max_workers=12) as ex:
    for i, rank in ex.map(one, range(len(rows))):
        ranks[i] = rank
        if (i + 1) % 12 == 0:
            print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

def summ(rs):
    a = np.array(rs); fin = [x for x in a if x < 888]
    return {"n": len(a), "@1": round(float((a <= 1).mean()), 3), "@5": round(float((a <= 5).mean()), 3),
            "@10": round(float((a <= 10).mean()), 3), "@100": round(float((a <= 100).mean()), 3),
            "median": float(np.median(fin)) if fin else None,
            "not_in_corpus": int((a == 888).sum()), "miss": int((a == 999).sum()),
            "query_err": int((a == 997).sum())}

zh = [i for i, r in enumerate(rows) if r[2] == "zh"]
en = [i for i, r in enumerate(rows) if r[2] == "en"]
out = {"total_n93": summ(ranks), "zh_orig_n31": summ([ranks[i] for i in zh[:31]]),
       "zh_colloquial_n31": summ([ranks[i] for i in zh[31:]]), "en_all_n31": summ([ranks[i] for i in en]),
       "_elapsed_s": round(time.time()-t0, 1),
       "_structure": "v93=31 目标×(zh原版/zh口语/en技术)；888=目标不在语料；997=查询重试后仍失败",
       "_rank_details": {f"{r[0]}|{r[2]}|{r[1][:20]}": ranks[i] for i, r in enumerate(rows)}}
json.dump(out, open(f"{LT}/logs/p2_final_eval.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if not k.startswith("_")}, ensure_ascii=False, indent=1), flush=True)
print("EVAL_DONE", flush=True)
