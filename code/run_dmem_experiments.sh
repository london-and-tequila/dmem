#!/bin/bash
# D-MEM Experiment Runner
# Usage: bash run_dmem_experiments.sh [exp1|exp2|exp3|exp4|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${MODEL:-gpt-4o-mini}"
BACKEND="${BACKEND:-openai}"
RATIO="${RATIO:-1.0}"
OUTPUT_DIR="results/dmem"
LOG_DIR="logs/dmem"
MAX_TURNS="${MAX_TURNS:-200}"

EXPERIMENT="${1:-all}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "=============================="
echo "D-MEM Experiments"
echo "Model: $MODEL"
echo "Backend: $BACKEND"
echo "Ratio: $RATIO"
echo "Experiment: $EXPERIMENT"
echo "=============================="

run_exp1() {
    echo ""
    echo ">>> Experiment 1: LoCoMo Main Benchmark (Table 1)"
    python3 dmem_test.py \
        --experiment exp1 \
        --model "$MODEL" \
        --backend "$BACKEND" \
        --ratio "$RATIO" \
        --output_dir "$OUTPUT_DIR" \
        --log_dir "$LOG_DIR"
    echo ">>> Experiment 1 complete."
}

run_exp2() {
    echo ""
    echo ">>> Experiment 2: Scalability & Cost (Figure 1 + Table 2)"
    python3 dmem_test.py \
        --experiment exp2 \
        --model "$MODEL" \
        --backend "$BACKEND" \
        --ratio "$RATIO" \
        --max_turns "$MAX_TURNS" \
        --include_noise_group \
        --output_dir "$OUTPUT_DIR" \
        --log_dir "$LOG_DIR"
    echo ">>> Experiment 2 complete."
}

run_exp3() {
    echo ""
    echo ">>> Experiment 3: Noise Robustness (Figure 2)"
    python3 dmem_test.py \
        --experiment exp3 \
        --model "$MODEL" \
        --backend "$BACKEND" \
        --ratio "$RATIO" \
        --output_dir "$OUTPUT_DIR" \
        --log_dir "$LOG_DIR"
    echo ">>> Experiment 3 complete."
}

run_exp4() {
    echo ""
    echo ">>> Experiment 4: Ablation Study (Table 3 + Figure 3)"
    python3 dmem_test.py \
        --experiment exp4 \
        --model "$MODEL" \
        --backend "$BACKEND" \
        --ratio "$RATIO" \
        --output_dir "$OUTPUT_DIR" \
        --log_dir "$LOG_DIR"
    echo ">>> Experiment 4 complete."
}

case "$EXPERIMENT" in
    exp1) run_exp1 ;;
    exp2) run_exp2 ;;
    exp3) run_exp3 ;;
    exp4) run_exp4 ;;
    all)
        run_exp1
        run_exp2
        run_exp3
        run_exp4
        ;;
    *)
        echo "Usage: $0 [exp1|exp2|exp3|exp4|all]"
        exit 1
        ;;
esac

echo ""
echo "=============================="
echo "All requested experiments complete."
echo "Results in: $OUTPUT_DIR/"
echo "Logs in:    $LOG_DIR/"
echo "=============================="

# Generate analysis figures
echo ""
echo ">>> Generating analysis plots..."
python3 analyze_dmem_results.py --input_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR/figures"
echo ">>> Analysis complete."
