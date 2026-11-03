import os
#!/usr/bin/env python3
# 纯本地机械扫（参照臂）：零 agent 迭代、零在线源——固定分面查询集 × 向量库，取并集
# 对照 A/B 两臂用：覆盖数、种子找回、去重条数、耗时
import json, sqlite3, sys, time, urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor

LT = os.environ.get("LITSEARCH_HOME", os.path.expanduser("~/litsearch"))
QD_URL = "http://localhost:6333"

# 分面查询集：主题空间的固定切片（与 aSDH 快速自发缩小相关；含中英混合以测库的语言鲁棒性）
FACETS = [
    "rapid spontaneous resolution of acute subdural hematoma",
    "spontaneous reduction acute subdural haematoma conservative management",
    "急性硬膜下血肿快速自发缩小 病例报告",
    "mechanism of rapid resolution subdural hematoma redistribution",
    "subdural hematoma transformation into hygroma rapid absorption",
    "acute subdural hematoma downward migration spinal subdural space",
    "rapid resolution of subdural hematoma in elderly patients",
    "infant pediatric subdural hematoma spontaneous rapid resolution",
    "CT features predictors of spontaneous resolution acute subdural hematoma",
    "traumatic acute subdural hematoma without surgery spontaneous improvement case series",
    "resolution of extradural epidural hematoma rapid spontaneous",
    "bilateral acute subdural hematoma rapid absorption conservative",
    "subdural hematoma resolution cerebrospinal fluid washout theory",
    "delayed resolution chronic subdural hematoma spontaneous",
    "acute subdural hematoma missed diagnosis spontaneous resolution",
]

def encode(q):
    req = urllib.request.Request("http://localhost:8004/embed",
        data=json.dumps({"texts": [q]}).encode(), headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=120).read())
    return np.array(d["vectors"][0], dtype=np.float32)

def search(args):
    i, q = args
    v = encode(q)
    from qdrant_client import QdrantClient
    QD = QdrantClient(url=QD_URL, prefer_grpc=True, timeout=120)
    r = QD.query_points("pubmed_full", query=v.tolist(), limit=100).points
    return q, [(p.payload["p"], p.score) for p in r]

t0 = time.time()
found = {}  # pmid -> (best_score, facet)
with ThreadPoolExecutor(max_workers=4) as ex:
    for q, hits in ex.map(search, list(enumerate(FACETS))):
        for pmid, s in hits:
            if pmid not in found or s > found[pmid][0]:
                found[pmid] = (s, q)
el = time.time() - t0

# 评分素材
seeds = [l.strip() for l in open(f"{LT}/goldset/asdh_seeds_pmids.txt") if l.strip()]
w1 = set()
# W1 baseline 在另一台机器，不在本机——跳过自动比对，由人工比对阶段做。
# 如需自动比对：export LITSEARCH_Q1_MEDLINE=/path/to/pubmed_W1.medline
_medline = os.environ.get("LITSEARCH_Q1_MEDLINE")
if _medline:
    try:
        with open(_medline, encoding="utf-8", errors="ignore") as f:
            for l in f:
                pass
    except Exception:
        pass
seed_hit = [s for s in seeds if s in found]
top_by_score = sorted(found.items(), key=lambda x: -x[1][0])

out = {
    "design": "纯本地机械扫：15 固定分面 × top-100 并集；零迭代零在线",
    "facets": len(FACETS), "elapsed_s": round(el, 1),
    "unique_pmids": len(found),
    "seeds_found": f"{len(seed_hit)}/{len(seeds)}",
    "seeds_found_list": seed_hit,
    "seeds_missed": [s for s in seeds if s not in found],
    "all_pmids": sorted(found.keys()),
    "top30_by_score": [{"pmid": p, "score": round(v[0], 4), "facet": v[1][:40]} for p, v in top_by_score[:30]],
}
json.dump(out, open(f"{LT}/logs/library_sweep.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("all_pmids", "top30_by_score", "seeds_found_list", "seeds_missed")}, ensure_ascii=False))
print("seeds:", len(seed_hit), "/", len(seeds), "missed:", out["seeds_missed"])
print("saved -> logs/library_sweep.json")
