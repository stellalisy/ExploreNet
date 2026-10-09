#!/bin/bash
# Start GenEval2 reward servers on this node, wait until they answer, then run
# the given command (e.g. `accelerate launch scripts/train.py ...`) and stop the
# servers when it exits.
#
# Environment:
#   GENEVAL2_PYTHON        python of the reward environment (requirements-reward.txt); default: python
#   GENEVAL2_NUM_SERVERS   servers to start, one per GPU index 0..N-1 (default: 1)
#   GENEVAL2_GPU_OFFSET    first GPU index used by the servers (default: 0)
#   GENEVAL2_PORT_BASE     first port (default: 18185)
#   GENEVAL2_MODEL_NAME    default: Qwen/Qwen3-VL-2B-Instruct
#   GENEVAL2_STARTUP_TIMEOUT seconds (default: 1200)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
GENEVAL2_PYTHON="${GENEVAL2_PYTHON:-python}"
GENEVAL2_NUM_SERVERS="${GENEVAL2_NUM_SERVERS:-1}"
GENEVAL2_GPU_OFFSET="${GENEVAL2_GPU_OFFSET:-0}"
GENEVAL2_PORT_BASE="${GENEVAL2_PORT_BASE:-18185}"
GENEVAL2_MODEL_NAME="${GENEVAL2_MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
GENEVAL2_STARTUP_TIMEOUT="${GENEVAL2_STARTUP_TIMEOUT:-1200}"

PIDS=()
URLS=()
for idx in $(seq 0 $((GENEVAL2_NUM_SERVERS - 1))); do
    port=$((GENEVAL2_PORT_BASE + idx))
    gpu=$((GENEVAL2_GPU_OFFSET + idx))
    URLS+=("http://127.0.0.1:${port}")
    echo "[GenEval2] starting server ${idx} on port ${port}, GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="$gpu" "$GENEVAL2_PYTHON" "$SCRIPT_DIR/geneval2_server.py" \
        --model-name "$GENEVAL2_MODEL_NAME" --port "$port" --device cuda &
    PIDS+=("$!")
done

cleanup() {
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT

export GENEVAL2_SERVER_URLS
GENEVAL2_SERVER_URLS=$(IFS=,; echo "${URLS[*]}")

ready=0
for i in $(seq 1 "$GENEVAL2_STARTUP_TIMEOUT"); do
    for pid in "${PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "FATAL: a GenEval2 server exited during startup" >&2
            exit 1
        fi
    done
    count=0
    for url in "${URLS[@]}"; do
        if curl -sf -o /dev/null --max-time 2 "$url"; then count=$((count + 1)); fi
    done
    if [ "$count" -eq "${#URLS[@]}" ]; then
        echo "[GenEval2] ${count} server(s) ready after ${i}s: ${GENEVAL2_SERVER_URLS}"
        ready=1
        break
    fi
    sleep 1
done
if [ "$ready" -ne 1 ]; then
    echo "FATAL: GenEval2 servers not ready within ${GENEVAL2_STARTUP_TIMEOUT}s" >&2
    exit 1
fi

"$@"
