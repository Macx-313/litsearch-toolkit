import os
#!/usr/bin/env python3
# P2 终评：93 条查询 vs pubmed_full（28.3M）
# id 映射：建库按 full_corpus 行序赋全局 id（half0 前半 + half1 后半），故 pmid→行号即 pmid→点 id
import json
import numpy as np
from qdrant_client import QdrantClient

LT = os.environ.get("LITSEARCH_HOME", os.path.expanduser("~/litsearch"))
QD = QdrantClient(url="http://localhost:6333", timeout=300)
COL = "pubmed_full"

rows = [l.rstrip("\n").split("\t") for l in open(f"{LT}/goldset/gold_queries_v93.tsv") if l.strip()]
targets = {r[0] for r in rows}
pmid2id = {}
H0 = 14168536
with open(f"{LT}/parsed/full_corpus.jsonl") as f:
    for lineno, line in enumerate(f):
        p = line.split('\"pmid\": ', 1)[1].split(',', 1)[0]
        if p in targets:
            pmid2id[p] = lineno if lineno < H0 else lineno  # 全局行号即全局 id
print(f"targets mapped: {len(pmid2id)}/{len(targets)}", flush=True)

qv = np.load(f"{LT}/goldset/gold_queries_v93_enc.npy")
assert qv.shape[0] == len(rows)
ranks = []
for i, (pmid, q, lang) in enumerate(rows):
    t = pmid2id.get(pmid)
    from qdrant_client import models as _m
    r = []
    for attempt in range(4):
        try:
            r = QD.query_points(COL, query=(qv[i] / np.linalg.norm(qv[i])).tolist(), limit=100).points
            break
        except Exception as e:
            if attempt == 3:
                print(f'query {i} failed: {str(e)[:100]}', flush=True)
            else:
                import time as _t; _t.sleep(20)
    ids = [p.id for p in r]
    ranks.append(ids.index(t) + 1 if t in ids else (999 if t is not None else 888))

def summ(rs):
    a = np.array(rs)
    fin = [x for x in a if x < 888]
    return {"n": len(a), "@1": round(float((a <= 1).mean()), 3), "@5": round(float((a <= 5).mean()), 3),
            "@10": round(float((a <= 10).mean()), 3), "@100": round(float((a <= 100).mean()), 3),
            "median": float(np.median(fin)) if fin else None,
            "not_in_corpus": int((a == 888).sum()), "miss": int((a == 999).sum())}

zh = [i for i, r in enumerate(rows) if r[2] == "zh"]
en = [i for i, r in enumerate(rows) if r[2] == "en"]
out = {
    "total_n93": summ(ranks),
    "zh_all_n62": summ([ranks[i] for i in zh]),
    "zh_orig_n31": summ([ranks[i] for i in zh[:31]]),
    "zh_colloquial_n31": summ([ranks[i] for i in zh[31:]]),
    "en_all_n31": summ([ranks[i] for i in en]),
    "neurocrit_subset_only": summ([ranks[i] for i, r in enumerate(rows) if r[0] in set(l.strip() for l in open(f'{LT}/encoded/p1_neurocrit.pmids.txt'))]),
    "_structure": "v93=31 目标×(zh原版/zh口语/en技术)；888=目标不在语料（3 篇 title-only 目标×3 形态=9 条）；997=查询重试后仍失败",
    "_rank_details": {f"{r[0]}|{r[2]}|{r[1][:20]}": ranks[i] for i, r in enumerate(rows)},
}
json.dump(out, open(f"{LT}/logs/p2_final_eval.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if not k.startswith('_')}, ensure_ascii=False, indent=1))
