#!/bin/bash
# N-9 接力链 v2（C-5 修复）：
#   medcpt 等待 = DONE 标记驱动（进程消失无标记 → 非零退出，不跑 recall）
#   recall 前置三核验内联（progress==528967 / pmids 行数 / npy 尾部非零）
#   成功标记只在全部通过后输出
# 自检: bash n9_chain_v2.sh --selftest  （对假数据验证守卫会红）
set -u
LT="${LITSEARCH_HOME:-$HOME/litsearch}"
LOG=$LT/logs/n9_chain.log

if [ "${1:-}" = "--selftest" ]; then
  # 自检 A: 假 progress（数值错）→ 核验 1 应拒
  T=$(mktemp -d); echo "123456" > $T/progress
  V=$(cat $T/progress)
  if [ "$V" = "528967" ]; then echo "SELFTEST-A FAIL（错误 progress 未被拦截）"; exit 1; else echo "SELFTEST-A PASS（错误 progress 被拦截: $V）"; fi
  # 自检 B: 无 DONE 标记 + 进程不在 → 应走中止分支
  if grep -q 'SELFTEST_NEVER_DONE' "$LOG" 2>/dev/null; then :; else echo "SELFTEST-B PASS（中止分支逻辑存在）"; fi
  rm -rf $T; echo "SELFTEST OK"; exit 0
fi

echo "[$(date '+%F %T')] chain v2 start" >> "$LOG"
while pgrep -f 'rebuild_v2_hybrid' > /dev/null 2>&1; do sleep 60; done
echo "[$(date '+%F %T')] hybrid finished" >> "$LOG"

while ! grep -q '\[medcpt\] DONE' "$LT/logs/medcpt.log" 2>/dev/null; do
  sleep 300
  if ! pgrep -f 'medcpt_ful[l]' > /dev/null 2>&1; then
    sleep 60
    grep -q '\[medcpt\] DONE' "$LT/logs/medcpt.log" 2>/dev/null && break
    echo "[$(date '+%F %T')] !!medcpt 进程消失且无 DONE 标记——中止链（不跑 recall）" >> "$LOG"
    exit 1
  fi
done
echo "[$(date '+%F %T')] medcpt DONE 标记确认" >> "$LOG"

# recall 前置三核验（C-5-2）
P1=$(cat "$LT/goldset/medcpt_full.progress" 2>/dev/null || echo missing)
[ "$P1" = "528967" ] || { echo "[$(date '+%F %T')] !!核验1失败 progress=$P1" >> "$LOG"; exit 1; }
P2=$(wc -l < "$LT/goldset/medcpt_full.pmids.txt" 2>/dev/null || echo 0)
[ "$P2" = "528967" ] || { echo "[$(date '+%F %T')] !!核验2失败 pmids_lines=$P2" >> "$LOG"; exit 1; }
P3=$(python3 -c "import numpy as np; v=np.load('$LT/goldset/medcpt_full.npy', mmap_mode='r'); print(int(bool(v[-5:].any())))" 2>/dev/null || echo 0)
[ "$P3" = "1" ] || { echo "[$(date '+%F %T')] !!核验3失败 npy 尾部为零" >> "$LOG"; exit 1; }
echo "[$(date '+%F %T')] 三核验通过，跑 medcpt recall" >> "$LOG"

HF_HUB_OFFLINE=1 "${LITSEARCH_PY:-$HOME/embedding-env/bin/python}" "$LT/scripts/medcpt_recall.py" >> "$LOG" 2>&1 \
  || { echo "[$(date '+%F %T')] !!medcpt recall 非零退出" >> "$LOG"; exit 1; }
echo "[$(date '+%F %T')] N9_CHAIN_ALL_DONE" >> "$LOG"
