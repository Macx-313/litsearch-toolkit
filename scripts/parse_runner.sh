#!/bin/bash
# 16 路并行解析 runner（重建符号链接 → 并行解析 → 等待 → 完成标记）
set -u
LT="${LITSEARCH_HOME:-$HOME/litsearch}"
LOG=$LT/logs/parse.log
echo "[$(date '+%F %T')] pass start" >> "$LOG"

# 重建符号链接（先清后建，含新下载完成的文件；旧 progress 保证不重复解析）
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
  python3 "$LT/scripts/parse_baseline.py" "$LT/chunks/c$i" "$LT/parsed/sub_$i.jsonl" "$LT/parsed/stat_$i.json" \
    --mesh-file "$LT/mesh/mesh_expanded.txt" >> "$LOG" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "[$(date '+%F %T')] PARSE_PASS_DONE" >> "$LOG"
