#!/bin/bash
# 窗口 2b 主作业（DP2 数据并行版）：双卡各跑完整 8B（TP1），各编码一半语料 → 合并 → 查询编码 → 建库+召回
# 依据：S3 实测单卡 37.42 docs/s > TP2 20.76 docs/s（8B 单卡放得下，TP 通信纯开销）
set -uo pipefail
LT="${LITSEARCH_HOME:-$HOME/litsearch}"
L=$LT/logs/master_p1.log
IMG=docker.m.daocloud.io/vllm/vllm-openai:v0.28.0
LOG(){ echo "[$(date '+%F %T')] $*" | tee -a "$L"; }
GATE(){ [ -s "$1" ] || { LOG "!!GATE FAIL: $1 不存在，中止"; exit 1; }; }
RUNENC(){ # $1=GPU $2=in $3=out_prefix $4=batch
  sudo docker run --rm --runtime=nvidia --shm-size 32g -e NVIDIA_VISIBLE_DEVICES=$1 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v $LT:/lit -v "${HF_CACHE:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
    --entrypoint python3 $IMG /lit/scripts/encode_corpus.py "$2" "$3" --tp 1 --batch "$4" 2>&1
}

# 语料切分（若未切）
if [ ! -f $LT/parsed/p1_half1.jsonl ]; then
  T=$(wc -l < $LT/parsed/p1_neurocrit.jsonl)
  H=$((T / 2))
  head -n $H $LT/parsed/p1_neurocrit.jsonl > $LT/parsed/p1_half0.jsonl
  tail -n $((T - H)) $LT/parsed/p1_neurocrit.jsonl > $LT/parsed/p1_half1.jsonl
  LOG "语料切分: $T -> $H + $((T-H))"
fi

# DP2：双容器并行（各 20s 显存稳定等待）
LOG "DP2 编码开始（GPU0: half0 / GPU1: half1）"
sleep 20
( RUNENC 0 /lit/parsed/p1_half0.jsonl /lit/encoded/p1_half0 512 >> $L 2>&1; echo "half0 rc=$?" >> $L ) &
P0=$!
( RUNENC 1 /lit/parsed/p1_half1.jsonl /lit/encoded/p1_half1 512 >> $L 2>&1; echo "half1 rc=$?" >> $L ) &
P1=$!
wait $P0; wait $P1
GATE $LT/encoded/p1_half0.report.json
GATE $LT/encoded/p1_half1.report.json
LOG "DP2 done: $(cat $LT/encoded/p1_half0.report.json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["docs_per_s"],"docs/s")') / $(cat $LT/encoded/p1_half1.report.json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["docs_per_s"],"docs/s")')"

# 合并向量
LOG "合并向量开始"
python3 - <<'EOF'
import numpy as np
v0 = np.load(""$LT"/encoded/p1_half0.npy")
v1 = np.load(""$LT"/encoded/p1_half1.npy")
p0 = [l.strip() for l in open(""$LT"/encoded/p1_half0.pmids.txt")]
p1 = [l.strip() for l in open(""$LT"/encoded/p1_half1.pmids.txt")]
assert v0.shape[1] == v1.shape[1], "dim mismatch"
np.save(""$LT"/encoded/p1_neurocrit.npy", np.concatenate([v0, v1]))
open(""$LT"/encoded/p1_neurocrit.pmids.txt", "w").write("\n".join(p0 + p1) + "\n")
print(f"merged: {v0.shape[0]}+{v1.shape[0]}={v0.shape[0]+v1.shape[0]} dim={v0.shape[1]}")
EOF
GATE $LT/encoded/p1_neurocrit.npy

# gold 查询编码（TP1 单卡，快）
LOG "gold 查询编码 开始"
sudo docker run --rm --runtime=nvidia --shm-size 32g -e NVIDIA_VISIBLE_DEVICES=0 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v $LT:/lit -v "${HF_CACHE:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
  --entrypoint python3 $IMG /lit/scripts/encode_queries.py /lit/goldset/gold_queries.tsv /lit/goldset/gold_queries_enc 2>&1 | grep -E '\[query-encode\]' | tee -a "$L"
GATE $LT/goldset/gold_queries_enc.npy

# 建库 + 召回（host CPU）
LOG "建库 + gold 召回 开始"
"${LITSEARCH_PY:-$HOME/embedding-env/bin/python}" $LT/scripts/build_and_recall.py /lit/encoded/p1_neurocrit /lit/parsed/p1_neurocrit.jsonl /lit/goldset/gold_queries_enc 2>&1 | tee -a "$L"
LOG "窗口 2b 全部完成（p1_gold_recall.json 已落盘）"
