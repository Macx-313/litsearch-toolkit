#!/bin/bash
set -u
LOG="$LT/logs"/auto_eval_watch.log
while true; do
  ST=$(curl -s --max-time 10 http://localhost:6333/collections/pubmed_full | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["status"])' 2>/dev/null || echo err)
  echo "[$(date '+%F %T')] status=$ST" >> "$LOG"
  if [ "$ST" = "green" ]; then
    echo "[$(date '+%F %T')] GREEN -> run final eval" >> "$LOG"
    "${LITSEARCH_PY:-$HOME/embedding-env/bin/python}" "$LT"/scripts/final_eval_v2.py >> "$LOG" 2>&1
    echo "[$(date '+%F %T')] FINAL_EVAL_DONE rc=$?" >> "$LOG"
    exit 0
  fi
  sleep 600
done
