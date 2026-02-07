#!/usr/bin/env bash
# =========================================================
# detect_oom_workers.sh
# ---------------------
# List all Docker containers named mn.worker*,
# show whether they were OOM-killed, their last exit code,
# host PID, and memory limit.
#
# Usage: sudo ./detect_oom_workers.sh
# =========================================================

set -euo pipefail

FILTER_NAME='mn.worker'     # Container name prefix to match

printf "%-12s %-10s %-8s %-8s %-8s\n" "Container" "OOMKilled" "Exit" "HostPID" "MemLim"

for cid in $(docker ps -q --filter "name=${FILTER_NAME}"); do
  # Strip the leading '/' from inspect output
  name=$(docker inspect --format '{{.Name}}' "$cid" | cut -c2-)

  # Container state JSON
  state=$(docker inspect --format '{{json .State}}' "$cid")

  # Extract fields
  oom=$(echo "$state"     | grep -o '"OOMKilled":[^,]*' | cut -d: -f2)
  exitcode=$(echo "$state"| grep -o '"ExitCode":[^,]*' | cut -d: -f2)
  hpid=$(docker inspect   --format '{{.State.Pid}}' "$cid")
  memlim=$(docker inspect --format '{{.HostConfig.Memory}}' "$cid")

  # Convert memory limit
  if [[ "$memlim" == "0" ]]; then
      memlim="unlimited"
  else
      memlim="$((memlim/1024/1024))M"
  fi

  printf "%-12s %-10s %-8s %-8s %-8s\n" "$name" "$oom" "$exitcode" "$hpid" "$memlim"
done
