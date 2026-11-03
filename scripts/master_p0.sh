#!/bin/bash
# 窗口 1 主作业：P0 全部实验（bf16 吞吐锚点 / FP8 比对 / 单卡锚点 / MedCPT 对照 / Qdrant 矩阵）
set -uo pipefail
LT="${LITSEARCH_HOME:-$HOME/litsearch}"
L=$LT/logs/master_p0.log
IMG=docker.m.daocloud.io/vllm/vllm-openai:v0.28.0
LOG(){ echo "[$(date '+%F %T')] $*" | tee -a "$L"; }
mkdir -p $LT/encoded $LT/logs

ENCODE(){ # $1=infile $2=out_prefix $3..=extra
  local IN=$1 OUT=$2; shift 2
  sudo docker run --rm --runtime=nvidia --shm-size 32g \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v $LT:/lit -v "${HF_CACHE:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
    --entrypoint python3 $IMG /lit/scripts/encode_corpus.py "$IN" "$OUT" "$@" 2>&1 | tee -a "$L" | grep -E '\[encode\] DONE|Error|error|Traceback' | tail -3
}
GATE(){ # 阶段熔断：上阶段 report 不存在即中止（runner trap 负责恢复 8001）
  [ -s "$1" ] || { LOG "!!GATE FAIL: $1 不存在，中止后续阶段"; exit 1; }
}

# S1: 11.8万条 bf16 TP2（吞吐锚点 + 主向量档）
LOG "S1 bf16 TP2 全量 11.8万条 开始"
ENCODE /lit/corpus/p0_corpus_100k.jsonl /lit/encoded/p0_100k_bf16 --tp 2 --batch 512
GATE $LT/encoded/p0_100k_bf16.report.json
LOG "S1 done: $(cat $LT/encoded/p0_100k_bf16.report.json 2>/dev/null)"

# S2: 前 1 万条 FP8 比对（cosine 判据 ≥0.99）
head -10000 $LT/corpus/p0_corpus_100k.jsonl > $LT/corpus/p0_10k.jsonl
LOG "S2 FP8 1万条 开始"
ENCODE /lit/corpus/p0_10k.jsonl /lit/encoded/p0_10k_fp8 --tp 2 --quant fp8 --batch 512
GATE $LT/encoded/p0_10k_fp8.report.json
LOG "S2 done: $(cat $LT/encoded/p0_10k_fp8.report.json 2>/dev/null)"
python3 $LT/scripts/p0_eval.py cosine /lit/encoded/p0_100k_bf16 /lit/encoded/p0_10k_fp8 2>&1 | tee -a "$L"

# S3: 单卡锚点（NVIDIA_VISIBLE_DEVICES=0 + TP1）
LOG "S3 bf16 TP1 单卡锚点 开始"
sudo docker run --rm --runtime=nvidia --shm-size 32g -e NVIDIA_VISIBLE_DEVICES=0 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v $LT:/lit -v "${HF_CACHE:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
  --entrypoint python3 $IMG /lit/scripts/encode_corpus.py /lit/corpus/p0_10k.jsonl /lit/encoded/p0_10k_bf16_1card --tp 1 --batch 512 2>&1 | tee -a "$L" | tail -2
GATE $LT/encoded/p0_10k_bf16_1card.report.json
LOG "S3 done: $(cat $LT/encoded/p0_10k_bf16_1card.report.json 2>/dev/null)"

# S4: MedCPT 对照（host CPU，1 万条）
LOG "S4 MedCPT 对照 开始"
HF_HUB_OFFLINE=1 "${LITSEARCH_PY:-$HOME/embedding-env/bin/python}" $LT/scripts/p0_eval.py medcpt $LT/corpus/p0_10k.jsonl /lit/encoded/p0_10k_medcpt 2>&1 | tee -a "$L" | tail -2
GATE $LT/encoded/p0_10k_medcpt.npy

# S5: Qdrant 量化矩阵（8 组 × 1 万条）
LOG "S5 Qdrant 矩阵 开始"
"${LITSEARCH_PY:-$HOME/embedding-env/bin/python}" $LT/scripts/qdrant_loadtest.py /lit/encoded/p0_100k_bf16 --n 10000 2>&1 | tee -a "$L"

LOG "窗口1全部完成（详见 logs/master_p0.log / p0_cosine.json / qdrant_loadtest.json / 各 report.json）"
