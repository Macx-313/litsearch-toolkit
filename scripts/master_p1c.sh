#!/bin/bash
# 窗口 2c 迷你窗口：仅补完 查询编码（TP1 单卡）→ 建库 → 召回
set -uo pipefail
LT="${LITSEARCH_HOME:-$HOME/litsearch}"
L=$LT/logs/master_p1.log
IMG=docker.m.daocloud.io/vllm/vllm-openai:v0.28.0
LOG(){ echo "[$(date '+%F %T')] $*" | tee -a "$L"; }
GATE(){ [ -s "$1" ] || { LOG "!!GATE FAIL: $1 不存在，中止"; exit 1; }; }

sleep 20
LOG "2c gold 查询编码 开始（TP1）"
sudo docker run --rm --runtime=nvidia --shm-size 32g -e NVIDIA_VISIBLE_DEVICES=0 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v $LT:/lit -v "${HF_CACHE:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
  --entrypoint python3 $IMG /lit/scripts/encode_queries.py /lit/goldset/gold_queries.tsv /lit/goldset/gold_queries_enc 2>&1 | tee -a "$L" | grep -E '\[query-encode\]|Error' | tail -3
GATE $LT/goldset/gold_queries_enc.npy

LOG "2c 建库 + gold 召回 开始"
"${LITSEARCH_PY:-$HOME/embedding-env/bin/python}" $LT/scripts/build_and_recall.py /lit/encoded/p1_neurocrit /lit/parsed/p1_neurocrit.jsonl /lit/goldset/gold_queries_enc 2>&1 | tee -a "$L"
LOG "窗口 2c 完成"
