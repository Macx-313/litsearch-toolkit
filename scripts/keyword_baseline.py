import os
#!/usr/bin/env python3
# 系统级对照：普通 agent 关键词检索（PubMed esearch 相关度排序）vs 向量库
# 臂设计：A=单发关键词（EN 查询直投）；B=两改写并集（全句 + 前8词压缩式，模拟勤奋 agent）
# 对照：向量臂 dense_en（pubmed_full，取自 p2_final_eval.json 的 en 行 rank_details）
import json, time, urllib.request, urllib.parse

LT = os.environ.get("LITSEARCH_HOME", os.path.expanduser("~/litsearch"))
rows = [l.rstrip("\n").split("\t") for l in open(f"{LT}/goldset/gold_queries_v93.tsv") if l.strip()]
# 每目标取 en 形态（agent 翻译后投关键词的等价物）
en_by_target = {}
for r in rows:
    if r[2] == "en":
        en_by_target[r[0]] = r[1]
targets = sorted(en_by_target, key=int)
print(f"targets with EN form: {len(targets)}", flush=True)

def esearch_ids(term, retmax=100):
    p = urllib.parse.urlencode({"db": "pubmed", "term": term, "retmax": retmax,
                                "sort": "relevance", "retmode": "json"})
    for i in range(4):
        try:
            j = json.loads(urllib.request.urlopen(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + p, timeout=30).read())
            return j["esearchresult"].get("idlist", [])
        except Exception as e:
            time.sleep(3)
    return []

STOP = {"of", "the", "and", "in", "for", "a", "to", "with", "after", "vs", "or"}
def compress(q, n=8):
    ws = [w for w in q.split() if w.lower() not in STOP]
    return " ".join(ws[:n])

res = {}
for t in targets:
    q = en_by_target[t]
    idsA = esearch_ids(q)
    idsB = esearch_ids(compress(q))
    res[t] = {
        "rankA": idsA.index(t) + 1 if t in idsA else 999,
        "rankB_union": (idsA + [x for x in idsB if x not in idsA]).index(t) + 1
                       if t in idsA or t in idsB else 999,
    }
    time.sleep(0.4)
json.dump(res, open(f"{LT}/logs/keyword_baseline.json", "w"), indent=1)

# 对照：向量臂 en 行（每目标的 en 查询 rank）
ev = json.load(open(f"{LT}/logs/p2_final_eval.json"))["_rank_details"]
vec_rank = {}
for r in rows:
    if r[2] == "en":
        vec_rank[r[0]] = ev[f"{r[0]}|en|{r[1][:20]}"]

import numpy as np
def summ(d, key=None):
    a = np.array([vec_rank[t] if key == "vec" else res[t][key] for t in targets])
    return {"@5": int((a <= 5).sum()), "@10": int((a <= 10).sum()),
            "@100": int((a <= 100).sum()), "miss": int((a == 999).sum()), "n": len(a)}
print("keyword 单发:", summ(res, "rankA"))
print("keyword 两改写并集:", summ(res, "rankB_union"))
print("向量 dense_en:", summ(None, "vec"))
# McNemar: 向量 vs 单发关键词 @100
from math import comb
vb = sum(1 for t in targets if vec_rank[t] <= 100 and res[t]["rankA"] > 100)
kb = sum(1 for t in targets if vec_rank[t] > 100 and res[t]["rankA"] <= 100)
nn = vb + kb
p = 1.0 if nn == 0 else min(1.0, 2 * sum(comb(nn, i) for i in range(min(vb, kb) + 1)) / 2 ** nn)
print(f"McNemar @100 向量 vs 关键词单发: b={vb} c={kb} p={p:.3f}")
