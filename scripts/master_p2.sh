#!/bin/bash
# P2 主作业：全量 28.3M 摘要 DP2 双卡编码（连续窗口，预计 ~4-5 天）
# 依赖：parsed/full_corpus.jsonl（parse_all_runner 产出合并后的全量）
# 产出：encoded/p2_half0 / p2_half1（向量+pmids+report，断点续跑）
set -uo pipefail
LT="${LITSEARCH_HOME:-$HOME/litsearch}"
L=$LT/logs/master_p2.log
IMG=docker.m.daocloud.io/vllm/vllm-openai:v0.28.0
LOG(){ echo "[$(date '+%F %T')] $*" | tee -a "$L"; }
GATE(){ [ -s "$1" ] || { LOG "!!GATE FAIL: $1"; exit 1; }; }

FULL=$LT/parsed/full_corpus.jsonl
GATE "$FULL"

# 对半切分（一次性）
if [ ! -f $LT/parsed/p2_half1.jsonl ]; then
  T=$(wc -l < "$FULL")
  H=$((T / 2))
  head -n $H "$FULL" > $LT/parsed/p2_half0.jsonl
  tail -n $((T - H)) "$FULL" > $LT/parsed/p2_half1.jsonl
  LOG "语料对半: $T -> $H + $((T-H))"
fi

RUNENC(){ # $1=GPU $2=in $3=prefix
  sudo docker run --rm --runtime=nvidia --shm-size 32g -e NVIDIA_VISIBLE_DEVICES=$1 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v $LT:/lit -v "${HF_CACHE:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
    --entrypoint python3 $IMG /lit/scripts/encode_corpus.py "$2" "$3" --tp 1 --batch 512 2>&1
}

LOG "P2 DP2 编码开始（GPU0: half0 / GPU1: half1，检查点续跑）"
sleep 20
( RUNENC 0 /lit/parsed/p2_half0.jsonl /lit/encoded/p2_half0 >> $L 2>&1; echo "half0 rc=$?" >> $L ) &
P0=$!
( RUNENC 1 /lit/parsed/p2_half1.jsonl /lit/encoded/p2_half1 >> $L 2>&1; echo "half1 rc=$?" >> $L ) &
P1=$!
wait $P0; wait $P1
GATE $LT/encoded/p2_half0.report.json
GATE $LT/encoded/p2_half1.report.json
LOG "P2 编码完成: $(cat $LT/encoded/p2_half0.report.json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["n_docs"], d["tokens_per_s"], "tok/s")') / $(cat $LT/encoded/p2_half1.report.json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["n_docs"], d["tokens_per_s"], "tok/s")')"
LOG "P2 编码阶段结束（建库与召回另行编排，不占窗口）"
