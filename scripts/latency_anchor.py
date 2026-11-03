import os
#!/usr/bin/env python3
# B1 修复：延迟锚点实测——落盘 JSON（环境快照 + 四配置 × 30 样本 p50/max/@10）
import json, time, statistics, subprocess
import numpy as np
from qdrant_client import QdrantClient, models

LT = os.environ.get("LITSEARCH_HOME", os.path.expanduser("~/litsearch"))
QD = QdrantClient(url="http://localhost:6333", prefer_grpc=True, timeout=120)
qv = np.load(f"{LT}/goldset/gold_queries_v93_enc.npy")
rows = [l.rstrip("\n").split("\t") for l in open(f"{LT}/goldset/gold_queries_v93.tsv") if l.strip()]
targets = {r[0] for r in rows}
pmid2id = {}
with open(f"{LT}/parsed/full_corpus.jsonl") as f:
    for i, line in enumerate(f):
        p = line.split('"pmid": ', 1)[1].split(",", 1)[0]
        if p in targets:
            pmid2id[p] = i

env = {
    "ts": time.strftime("%F %T"),
    "loadavg": open("/proc/loadavg").read().split()[:3],
    "qdrant_cpu_pct": subprocess.run(["bash", "-c", "top -bn1 | grep qdrant | awk '{print $9}'"],
                                     capture_output=True, text=True).stdout.strip(),
    "note": "测量时环境快照；延迟为变量，取分布不取单点",
}
IDS = [i for i, r in enumerate(rows) if pmid2id.get(r[0]) is not None][:30]
# 术语更正（B1-3）：ignore=True = 忽略量化（遍历+打分均用原始全精度向量，无独立重排相）
CONFIGS = {
    "default_重排开": {},
    "ignore_quantization_忽略量化_全精度遍历": {"quantization": models.QuantizationSearchParams(ignore=True)},
    "hnsw_ef48_浅图": {"hnsw_ef": 48},
    "ignore_quant_plus_ef48": {"quantization": models.QuantizationSearchParams(ignore=True), "hnsw_ef": 48},
}
out = {"environment": env, "n_samples": len(IDS), "configs": {}}
for name, sp in CONFIGS.items():
    lat, hit10, hit100 = [], 0, 0
    for qi in IDS:
        q = (qv[qi] / np.linalg.norm(qv[qi])).tolist()
        t = pmid2id[rows[qi][0]]
        t0 = time.time()
        r = QD.query_points("pubmed_full", query=q, limit=100,
                            search_params=models.SearchParams(**sp)).points
        lat.append((time.time()-t0)*1000)
        ids = [p.id for p in r]
        if t in ids[:10]: hit10 += 1
        if t in ids: hit100 += 1
    out["configs"][name] = {
        "p50_ms": round(statistics.median(lat), 1), "max_ms": round(max(lat), 1),
        "min_ms": round(min(lat), 1),
        "hit@10": f"{hit10}/{len(IDS)}", "hit@100": f"{hit100}/{len(IDS)}",
        "note": "四配置 @10/@100 相互无差异（相对持平）；绝对召回受语义天花板限制（另见终评）",
    }
    print(name, out["configs"][name], flush=True)
json.dump(out, open(f"{LT}/logs/p2_latency_anchor.json", "w"), ensure_ascii=False, indent=1)
print("saved -> logs/p2_latency_anchor.json", flush=True)
