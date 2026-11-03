#!/bin/bash
# 窗口 2 主作业：S3 单卡锚点补跑 → 52.9 万篇 bf16 编码 → gold 查询编码 → 建库 + 召回测试
set -uo pipefail
LT="${LITSEARCH_HOME:-$HOME/litsearch}"
L=$LT/logs/master_p1.log
IMG=docker.m.daocloud.io/vllm/vllm-openai:v0.28.0
LOG(){ echo "[$(date '+%F %T')] $*" | tee -a "$L"; }
GATE(){ [ -s "$1" ] || { LOG "!!GATE FAIL: $1 不存在，中止"; exit 1; }; }
mkdir -p "$LT/logs"

# S3: 单卡锚点（20s 显存释放等待——上轮教训）
LOG "S3 单卡锚点 开始"
sleep 20
sudo docker run --rm --runtime=nvidia --shm-size 32g -e NVIDIA_VISIBLE_DEVICES=0 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v $LT:/lit -v "${HF_CACHE:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
  --entrypoint python3 $IMG /lit/scripts/encode_corpus.py /lit/corpus/p0_10k.jsonl /lit/encoded/p0_10k_bf16_1card --tp 1 --batch 512 2>&1 | tail -2 | tee -a "$L"
GATE $LT/encoded/p0_10k_bf16_1card.report.json
LOG "S3 done: $(cat $LT/encoded/p0_10k_bf16_1card.report.json)"

# 主编码：529,091 篇 bf16 TP2
LOG "P1 编码 52.9 万篇 开始"
sudo docker run --rm --runtime=nvidia --shm-size 32g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v $LT:/lit -v "${HF_CACHE:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
  --entrypoint python3 $IMG /lit/scripts/encode_corpus.py /lit/parsed/p1_neurocrit.jsonl /lit/encoded/p1_neurocrit --tp 2 --batch 512 2>&1 | tee -a "$L" | grep -E '\[encode\] (DONE|[0-9]+/)' | tail -20
GATE $LT/encoded/p1_neurocrit.report.json
LOG "P1 编码 done: $(cat $LT/encoded/p1_neurocrit.report.json)"

# gold 查询编码（22 条，instruction 格式，同模型同精度）
LOG "gold 查询编码 开始"
sudo docker run --rm --runtime=nvidia --shm-size 32g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v $LT:/lit -v "${HF_CACHE:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
  --entrypoint python3 $IMG /lit/scripts/encode_queries.py /lit/goldset/gold_queries.tsv /lit/goldset/gold_queries_enc 2>&1 | grep -E '\[query-encode\]' | tee -a "$L"
GATE $LT/goldset/gold_queries_enc.npy

# 建库 + 召回（host CPU）
LOG "建库 + gold 召回 开始"
"${LITSEARCH_PY:-$HOME/embedding-env/bin/python}" $LT/scripts/build_and_recall.py /lit/encoded/p1_neurocrit /lit/parsed/p1_neurocrit.jsonl /lit/goldset/gold_queries_enc 2>&1 | tee -a "$L"
LOG "窗口 2 全部完成（p1_gold_recall.json 已落盘）"
