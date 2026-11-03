#!/bin/bash
# B 段 GPU 窗口 runner v3 —— Step 0-5
# v3 修复（WB 终验 H-1/H-7）：
#   H-1: 总结行"驱动项"动态拼装——只列真实触发项，不再硬编码三件套；中止路径写明具体断言
#   H-7: 8001 health 等待上限 240s→900s；超时与健康失败分开报
# 保留 v2 全部能力（RC 传导/C-4 溯源/巡检告警）
set -uo pipefail
JOB="${1:?用法: litsearch_b_window.sh \"<job command>\"}"
LT="${LITSEARCH_HOME:-$HOME/litsearch}"
L=$LT/logs/window.log
META=$LT/logs/window_meta.jsonl
RC_FILE=$LT/logs/last_job.rc
mkdir -p "$LT/logs" "$LT/jobs"
FAIL=0
RC=0
ABORT_REASON=""
LOG(){ echo "[$(date '+%F %T')] $*" | tee -a "$L"; }

restore(){
  LOG "Step 5 恢复开始"
  tmux kill-session -t litpatrol 2>/dev/null
  sudo docker start vllm-nvfp4 && LOG "容器已启动" || { FAIL=1; ABORT_REASON="$ABORT_REASON 容器启动失败;"; }
  sudo systemctl start vllm-watchdog.timer && LOG "watchdog timer 已恢复" || { FAIL=1; ABORT_REASON="$ABORT_REASON timer 恢复失败;"; }
  CODE=000; WAITED=0
  for i in $(seq 1 90); do   # H-7: 上限 900s
    CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:8001/v1/models)
    [ "$CODE" = "200" ] && break
    sleep 10; WAITED=$((WAITED+10))
  done
  if [ "$CODE" = "200" ]; then
    LOG "8001 health=200（等待 ${WAITED}s）"
  else
    FAIL=1; ABORT_REASON="$ABORT_REASON 8001 等待 ${WAITED}s 后仍 $CODE（超时与失败已区分——超时可能是加载慢，非健康失败）;"
  fi
  if [ -s "$LT/logs/patrol_alerts.log" ]; then
    FAIL=1; ABORT_REASON="$ABORT_REASON 巡检告警(patrol_alerts.log 非空);"
  fi
  [ "$RC" = "0" ] || { FAIL=1; ABORT_REASON="$ABORT_REASON 作业退出码=$RC;"; }
  if [ "$FAIL" = "0" ]; then
    LOG "窗口结束 FAIL=0（全部通过：作业 rc=0、恢复断言全绿、无巡检告警）"
  else
    LOG "窗口结束 FAIL=1（实际触发：${ABORT_REASON:- 恢复阶段断言失败}）"
  fi
}
trap restore EXIT

LOG "Step 0 前置断言"
timeout 15 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tee -a "$L"
tmux ls 2>/dev/null | tee -a "$L"

LOG "Step 1 禁用 watchdog（整个窗口期）"
sudo systemctl stop vllm-watchdog.timer
WD=$(systemctl is-active vllm-watchdog.timer)
LOG "watchdog is-active=$WD（断言 inactive）"
[ "$WD" = "inactive" ] || { LOG "!!中止：Step1 watchdog 停不掉（is-active=$WD）"; ABORT_REASON="watchdog 停不掉"; FAIL=1; exit 1; }

LOG "Step 2 停 vllm-nvfp4"
POL=$(sudo docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' vllm-nvfp4)
LOG "RestartPolicy=$POL（断言 unless-stopped）"
[ "$POL" = "unless-stopped" ] || { LOG "!!中止：RestartPolicy=$POL 非 unless-stopped"; FAIL=1; ABORT_REASON="RestartPolicy 异常"; exit 1; }
sudo docker stop vllm-nvfp4
sleep 8
M0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
LOG "max_gpu_mem_used=${M0}MiB（断言 <500）"
[ "$M0" -lt 500 ] || { LOG "!!中止：显存未释放（${M0}MiB ≥500）"; FAIL=1; ABORT_REASON="显存断言拦截（${M0}MiB）"; exit 1; }

META_JSON=$(python3 - "$JOB" <<'PYEOF'
import sys, json, hashlib, time, os, re, shutil
job = sys.argv[1]
entry = {"ts_start": time.strftime("%F %T"), "job": job,
         "owner": os.environ.get("LITSEARCH_OWNER", "zcode")}
m = re.search(r"(/\S+\.sh)", job)
if m and os.path.exists(m.group(1)):
    js = m.group(1)
    entry["job_script"] = js
    entry["job_script_md5"] = hashlib.md5(open(js, "rb").read()).hexdigest()
    dst = ""$LT"/jobs/" + time.strftime("%Y%m%d_%H%M%S_") + os.path.basename(js)
    shutil.copy2(js, dst)
    entry["job_copy"] = dst
print(json.dumps(entry, ensure_ascii=False))
PYEOF
)
echo "$META_JSON" >> "$META"
LOG "Step 3 启动作业: $JOB"
LOG "RUN VERSION: $(echo "$META_JSON" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("job_script_md5","inline"), "->", d.get("job_copy","inline"))')"

( bash -c "$JOB" 2>&1; echo $? > "$RC_FILE" ) | tee -a "$L" &
JOB_PID=$!
RC=0
tmux kill-session -t litpatrol 2>/dev/null
rm -f "$LT/logs/patrol_alerts.log"
tmux new-session -d -s litpatrol "while true; do PATROL_IN_WINDOW=1 bash $LT/scripts/litsearch_b_patrol.sh || echo \"[\$(date '+%F %T')] !!PATROL_ASSERTION_FAILED rc=\$?\" >> $LT/logs/patrol_alerts.log; sleep 1800; done"
LOG "巡检会话 litpatrol 已启动（30min 周期）"
wait "$JOB_PID"
RC=$(cat "$RC_FILE" 2>/dev/null || echo 1)
LOG "作业退出码=$RC"
