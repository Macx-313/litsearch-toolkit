#!/bin/bash
# vLLM 守卫 v4：时间戳比较法识别进行中窗口（对夭折窗口的账目不平衡免疫）
set -u
LOG="$LT/logs"/vllm_guard.log
WL="$LT/logs"/window.log
TS=$(date '+%F %T')
RUNNING=$(sudo -n docker inspect -f '{{.State.Running}}' vllm-nvfp4 2>/dev/null || echo unknown)
[ "$RUNNING" = "true" ] && exit 0
LAST_S3=$(grep 'Step 3 启动作业' "$WL" 2>/dev/null | tail -1 | grep -oE '^\[[^]]+\]' | tr -d '[]')
LAST_CLOSE=$(grep '窗口结束' "$WL" 2>/dev/null | tail -1 | grep -oE '^\[[^]]+\]' | tr -d '[]')
if [ -n "$LAST_S3" ] && [ "$LAST_S3" \> "$LAST_CLOSE" ]; then
  AGE_OK=$(python3 -c "
from datetime import datetime
try:
    d=(datetime.now()-datetime.strptime('$LAST_S3','%Y-%m-%d %H:%M:%S')).total_seconds()
    print(1 if d < 86400 else 0)
except Exception:
    print(1)" 2>/dev/null || echo 1)
  if [ "$AGE_OK" = "1" ]; then
    echo "[$TS] 容器停机但 runner 窗口进行中（Step3@$LAST_S3 晚于末次结束@$LAST_CLOSE）——守卫让位" >> "$LOG"
    exit 0
  fi
  echo "[$TS] note: Step3 超 24h 未闭合（疑似崩溃窗口），守卫继续履职" >> "$LOG"
fi
echo "[$TS] !!发现停机（无窗口、GPU 无人）→ 自动恢复" >> "$LOG"
sudo -n docker start vllm-nvfp4 >> "$LOG" 2>&1
sudo -n systemctl start vllm-watchdog.timer >> "$LOG" 2>&1
echo "[$TS] 恢复动作完成 rc=$?" >> "$LOG"
