#!/usr/bin/env bash
# Multi-model concurrent evaluation across noise levels.
# Usage: bash scripts/run_all.sh [--models "m1 m2"] [--methods "d-mem a-mem"] [--noise_ratios "0 25 50 75"] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Configuration (defaults, overridable via CLI)
# ---------------------------------------------------------------------------
MODELS=""
METHODS="d-mem a-mem"
NOISE_RATIOS="0 25 50 75"
LOG_DIR="logs"
DRY_RUN=false

# vLLM / SGLang ports
PORT_A=30000
PORT_B=30001
GPUS_3B="0"
GPUS_1B="1"
TP=1
HEALTH_TIMEOUT=300

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models) MODELS="$2"; shift 2;;
        --methods) METHODS="$2"; shift 2;;
        --noise_ratios) NOISE_RATIOS="$2"; shift 2;;
        --dry-run) DRY_RUN=true; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# Default API models if --models not specified
if [[ -z "$MODELS" ]]; then
    MODELS="gpt-4o-mini gpt-4o claude-3-haiku-20240307 claude-3-5-haiku-20241022"
fi

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Model → backend mapping
# ---------------------------------------------------------------------------
get_backend() {
    case "$1" in
        gpt-*) echo "openai";;
        claude-*) echo "litellm";;
        *) echo "sglang";;
    esac
}

# ---------------------------------------------------------------------------
# Helpers (reused from code/run_all_experiments.sh)
# ---------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

kill_port() {
    local port=$1
    local pids
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        log "Killing processes on port $port: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

wait_for_server() {
    local port=$1
    local name=$2
    local elapsed=0
    log "Waiting for $name on port $port ..."
    while ! curl -s "http://localhost:${port}/health" >/dev/null 2>&1 && \
          ! curl -s "http://localhost:${port}/v1/models" >/dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [[ $elapsed -ge $HEALTH_TIMEOUT ]]; then
            log "ERROR: $name on port $port did not start within ${HEALTH_TIMEOUT}s"
            return 1
        fi
    done
    log "$name ready on port $port (${elapsed}s)"
}

launch_vllm() {
    local model=$1
    local port=$2
    local tp=$3
    local gpus=$4
    local logfile=$5
    log "Launching vLLM: $model on port $port (GPUs=$gpus, TP=$tp)"
    if $DRY_RUN; then
        log "[DRY-RUN] Would launch vLLM for $model"
        return
    fi
    CUDA_VISIBLE_DEVICES=$gpus VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS \
        nohup python -m vllm.entrypoints.openai.api_server \
        --model "$model" --port "$port" \
        --tensor-parallel-size "$tp" \
        --dtype float16 \
        --gpu-memory-utilization 0.90 \
        > "$logfile" 2>&1 &
    echo $!
}

run_eval() {
    # Wrapper that runs scripts/run_eval.py with nohup
    local model=$1
    local backend=$2
    local method=$3
    local noise=$4
    local extra_args="${5:-}"
    local tag
    tag=$(echo "${model}" | tr '/' '-')_${method}_n${noise}
    local logfile="${LOG_DIR}/eval_${tag}.log"

    log "Starting eval: model=$model method=$method noise=$noise backend=$backend"
    if $DRY_RUN; then
        log "[DRY-RUN] Would run eval for $tag"
        return
    fi
    nohup python scripts/run_eval.py \
        --model "$model" --backend "$backend" \
        --method "$method" --noise_ratio "$noise" \
        --wandb $extra_args \
        > "$logfile" 2>&1 &
}

# ---------------------------------------------------------------------------
# Phase 1: API models (can all run in parallel)
# ---------------------------------------------------------------------------
log "========== Phase 1: API Models =========="

API_PIDS=()
for model in $MODELS; do
    backend=$(get_backend "$model")
    for noise in $NOISE_RATIOS; do
        for method in $METHODS; do
            run_eval "$model" "$backend" "$method" "$noise"
            API_PIDS+=($!)
        done
    done
done

# ---------------------------------------------------------------------------
# Phase 2: vLLM models (sequential server launches, parallel evals per server)
# ---------------------------------------------------------------------------
log "========== Phase 2: vLLM Models =========="

if ! $DRY_RUN; then
    # --- Qwen ---
    kill_port $PORT_A
    QWEN_MODEL="Qwen/Qwen2.5-3B-Instruct"
    QWEN_PID=$(launch_vllm "$QWEN_MODEL" $PORT_A $TP "$GPUS_3B" "$LOG_DIR/vllm_qwen.log")
    wait_for_server $PORT_A "Qwen2.5-3B"

    QWEN_PIDS=()
    for noise in $NOISE_RATIOS; do
        for method in $METHODS; do
            run_eval "$QWEN_MODEL" "sglang" "$method" "$noise" \
                "--sglang_port $PORT_A"
            QWEN_PIDS+=($!)
        done
    done

    # Wait for all Qwen evals
    for pid in "${QWEN_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    log "Qwen evals complete, shutting down server"
    kill_port $PORT_A

    # --- Llama ---
    LLAMA_MODEL="meta-llama/Llama-3.2-3B-Instruct"
    LLAMA_PID=$(launch_vllm "$LLAMA_MODEL" $PORT_A $TP "$GPUS_3B" "$LOG_DIR/vllm_llama.log")
    wait_for_server $PORT_A "Llama-3.2-3B"

    LLAMA_PIDS=()
    for noise in $NOISE_RATIOS; do
        for method in $METHODS; do
            run_eval "$LLAMA_MODEL" "sglang" "$method" "$noise" \
                "--sglang_port $PORT_A"
            LLAMA_PIDS+=($!)
        done
    done

    for pid in "${LLAMA_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    log "Llama evals complete, shutting down server"
    kill_port $PORT_A
fi

# ---------------------------------------------------------------------------
# Wait for API model evals
# ---------------------------------------------------------------------------
log "Waiting for API model evaluations to complete..."
for pid in "${API_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "=========================================="
log "All evaluations complete!"
log "Results: results/dmem/"
log "Graphs:  results/dmem/graphs/"
log "Logs:    $LOG_DIR/"
log ""
log "Next steps:"
log "  python scripts/analyze_graph.py --graph_dir results/dmem/graphs/"
log "=========================================="
