#!/bin/bash
set -u
LOG="$LT/logs"/green_watch.log
while true; do
  ST=$(curl -s --max-time 10 http://localhost:6333/collections/pubmed_full | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["status"])' 2>/dev/null || echo err)
  echo "[$(date '+%F %T')] status=$ST" >> "$LOG"
  if [ "$ST" = "green" ]; then
    echo "[$(date '+%F %T')] GREEN! 跑延迟实测" >> "$LOG"
    "${LITSEARCH_PY:-$HOME/embedding-env/bin/python}" - <<'PYEOF' >> "$LOG" 2>&1
import time, statistics
import numpy as np
from qdrant_client import QdrantClient
QD = QdrantClient(url="http://localhost:6333", prefer_grpc=True, timeout=60)
qv = np.load(""$LT"/goldset/gold_queries_v93_enc.npy")
lat = []
for i in range(10):
    q = (qv[i] / np.linalg.norm(qv[i])).tolist()
    t0 = time.time()
    QD.query_points("pubmed_full", query=q, limit=100)
    lat.append((time.time()-t0)*1000)
print(f"GREEN_LATENCY: n=10 p50={statistics.median(lat):.1f}ms max={max(lat):.1f}ms (28.3M 点库)")
PYEOF
    exit 0
  fi
  sleep 300
done
