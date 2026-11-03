#!/bin/bash
# N-9 接力链 v3（取代 v1/v2，H-2/H-3 修复）
#   DONE 标记驱动 + 三核验内联（--check-only 可单独调用，LT 可注入）
#   自检双向：错误数据必须红（真红）、正确数据必须绿（真绿）——不再是恒 PASS 空转
# 用法: n9_chain.sh [--check-only | --selftest]
set -u
LT="${LITSEARCH_HOME:-$HOME/litsearch}"
LOG=$LT/logs/n9_chain.log

check_preconditions() {  # 三核验：全过返回 0，否则返回 1 并打印失败项
  local P1 P2 P3
  P1=$(cat "$LT/goldset/medcpt_full.progress" 2>/dev/null || echo missing)
  [ "$P1" = "528967" ] || { echo "核验1失败 progress=$P1"; return 1; }
  P2=$(wc -l < "$LT/goldset/medcpt_full.pmids.txt" 2>/dev/null | tr -d ' ' || echo 0)
  [ "$P2" = "528967" ] || { echo "核验2失败 pmids_lines=$P2"; return 1; }
  P3=$(python3 -c "import numpy as np; v=np.load('$LT/goldset/medcpt_full.npy', mmap_mode='r'); print(int(bool(v[-5:].any())))" 2>/dev/null || echo 0)
  [ "$P3" = "1" ] || { echo "核验3失败 npy 尾部为零"; return 1; }
  echo "三核验通过"
  return 0
}

if [ "${1:-}" = "--check-only" ]; then
  check_preconditions; exit $?
fi

if [ "${1:-}" = "--selftest" ]; then
  F=$(mktemp -d); mkdir -p "$F/goldset" "$F/logs"
  python3 -c "import numpy as np; np.save('$F/goldset/medcpt_full.npy', np.ones((10,768),dtype=np.float32))"
  # 红向：错误 progress 必须被拦
  echo 123456 > "$F/goldset/medcpt_full.progress"; echo x > "$F/goldset/medcpt_full.pmids.txt"
  if LT_OVERRIDE="$F" bash "$0" --check-only >/dev/null 2>&1; then
    echo "SELFTEST-A FAIL：错误 progress 未被拦截（恒绿灯！）"; rm -rf "$F"; exit 1
  fi
  echo "SELFTEST-A PASS（错误 progress 真红）"
  # 绿向：正确数据必须通过（造 528,967 行 pmids 用轻量法：wc -l 断言的是行数，写 528967 个换行）
  echo 528967 > "$F/goldset/medcpt_full.progress"
  python3 -c "open('$F/goldset/medcpt_full.pmids.txt','w').write('\n'*528967)"
  if LT_OVERRIDE="$F" bash "$0" --check-only >/dev/null 2>&1; then
    echo "SELFTEST-B PASS（正确数据真绿）"
  else
    echo "SELFTEST-B FAIL：正确数据被拦（噪音源！）"; rm -rf "$F"; exit 1
  fi
  rm -rf "$F"; echo "SELFTEST OK（双向）"; exit 0
fi

echo "[$(date '+%F %T')] chain v3 start" >> "$LOG"
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

if ! check_preconditions >> "$LOG" 2>&1; then
  echo "[$(date '+%F %T')] !!三核验未过——不跑 recall" >> "$LOG"; exit 1
fi
HF_HUB_OFFLINE=1 "${LITSEARCH_PY:-$HOME/embedding-env/bin/python}" "$LT/scripts/medcpt_recall.py" >> "$LOG" 2>&1 \
  || { echo "[$(date '+%F %T')] !!medcpt recall 非零退出" >> "$LOG"; exit 1; }
echo "[$(date '+%F %T')] N9_CHAIN_ALL_DONE" >> "$LOG"
