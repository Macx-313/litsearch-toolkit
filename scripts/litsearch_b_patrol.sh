#!/bin/bash
# B 段巡检 v2.1（H-9）：watchdog 断言仅在窗口内生效（PATROL_IN_WINDOW=1 由 runner 注入）；
# 窗口外手跑只记录不判红（20:37"实弹红灯"实际踩的是本断言的无条件性，非被守护逻辑）
set -uo pipefail
L="$LT/logs"/patrol.log
TS(){ date '+%F %T'; }
FAIL=0
{
echo "=== patrol $(TS) (in_window=${PATROL_IN_WINDOW:-0}) ==="
timeout 15 nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used --format=csv,noheader \
  || { echo "!!ALERT nvidia-smi 超时/失败"; FAIL=1; }
timeout 10 df -h /home | tail -1 || { echo "!!ALERT df 超时"; FAIL=1; }
WD=$(timeout 5 systemctl is-active vllm-watchdog.timer 2>/dev/null); WD_RC=$?
[ -z "$WD" ] && WD="query-timeout"
echo "watchdog_timer=$WD"
if [ "${PATROL_IN_WINDOW:-0}" = "1" ]; then
  [ "$WD" = "inactive" ] || { echo "!!ALERT 窗口期内 watchdog 非 inactive——静默复活风险，立即人工核查"; FAIL=1; }
else
  [ "$WD" = "inactive" ] && echo "note: 非窗口期但 watchdog inactive（若非窗口编排所致请核查）"
fi
CODE=$(timeout 5 curl -s -o /dev/null -w '%{http_code}' http://localhost:6333/readyz || echo timeout)
echo "qdrant_readyz=$CODE"
[ "$CODE" = "200" ] || { echo "!!ALERT qdrant readyz 异常: $CODE"; FAIL=1; }
for p in "$LT"/encoded/*.progress; do
  [ -f "$p" ] && echo "progress $(basename "$p" .progress): $(tail -1 "$p")"
done
pgrep -af 'encode_corpus' | head -2 || echo "note: 无 encode_corpus 进程（非编码期正常）"
echo "patrol_fail=$FAIL"
} >> "$L" 2>&1
exit $FAIL
