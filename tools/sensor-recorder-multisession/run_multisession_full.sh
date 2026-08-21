#!/usr/bin/env bash
# Run independent single-session reconstructions, then merge them into a joint VI-Map.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output=""
jobs=1
force=0
sessions=()

usage() {
  cat <<'EOF'
usage: process.sh multisession --data <SensorRecorder folder> [--data ...] --output <joint-result> [--jobs N] [--force]

Runs each input through `process.sh sfm`, using at most N concurrent workers,
then runs cross-session loop closure, PGO, observation fusion, and joint VI-BA.
Single-session results are stored in <joint-result>/sessions/.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) sessions+=("$2"); shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --jobs) jobs="$2"; shift 2 ;;
    --force) force=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown multisession argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$output" && "${#sessions[@]}" -ge 2 ]] || { usage >&2; exit 2; }
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs must be a positive integer" >&2; exit 2; }

output="$(realpath -m "$output")"
mkdir -p "$output/sessions" "$output/logs"

run_session() {
  local data="$1" id="$2" log="$3"
  local args=(sfm --data "$data" --output "$output/sessions/${id}_sfm")
  [[ "$force" -eq 1 ]] && args+=(--force)
  echo "[run] session=$id log=$log"
  "${repo_dir}/process.sh" "${args[@]}" >"$log" 2>&1
}

result_sessions=()
batch_pids=()
batch_ids=()
failed=0
wait_batch() {
  local index pid id
  for index in "${!batch_pids[@]}"; do
    pid="${batch_pids[$index]}"; id="${batch_ids[$index]}"
    if wait "$pid"; then
      echo "[done] session=$id"
    else
      echo "[error] session=$id; see $output/logs/${id}.log" >&2
      failed=1
    fi
  done
  batch_pids=()
  batch_ids=()
}

for data in "${sessions[@]}"; do
  data="$(realpath "$data")"
  [[ -d "$data" ]] || { echo "missing session data: $data" >&2; exit 2; }
  id="$(basename "$data")"
  result_sessions+=("$output/sessions/${id}_sfm")
  run_session "$data" "$id" "$output/logs/${id}.log" &
  batch_pids+=("$!")
  batch_ids+=("$id")
  if [[ "${#batch_pids[@]}" -ge "$jobs" ]]; then
    wait_batch
  fi
done
wait_batch
[[ "$failed" -eq 0 ]] || exit 1

merge_args=(--output "$output")
for result in "${result_sessions[@]}"; do merge_args+=(--session "$result"); done
exec "${repo_dir}/tools/sensor-recorder-multisession/run_multisession.sh" "${merge_args[@]}"
