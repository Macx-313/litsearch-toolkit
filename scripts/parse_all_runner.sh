#!/bin/bash
# P2 全量解析 runner：先清后建分片（16 路复用 chunks/）→ 并行 → 完成标记
set -u
LT="${LITSEARCH_HOME:-$HOME/litsearch}"
LOG=$LT/logs/parse_all.log
echo "[$(date '+%F %T')] full-parse start" >> "$LOG"
for i in $(seq 0 15); do
  mkdir -p "$LT/chunks/c$i"
  rm -f "$LT"/chunks/c$i/*.gz
  n=0
  for f in "$LT"/pubmed/baseline/pubmed26n*.xml.gz; do
    [ $((n % 16)) -eq $i ] && ln -sf "$f" "$LT/chunks/c$i/"
    n=$((n+1))
  done
done
pids=()
for i in $(seq 0 15); do
  python3 "$LT/scripts/parse_all.py" "$LT/chunks/c$i" "$LT/parsed/all_$i.jsonl" "$LT/parsed/allstat_$i.json" >> "$LOG" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "[$(date '+%F %T')] FULL_PARSE_DONE" >> "$LOG"
