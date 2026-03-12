"""
D-MEM result analysis: generate tables (stdout) and matplotlib figures.
Reads JSON outputs from dmem_test.py experiments 1–4.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available. Skipping figure generation.")


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Table 1: Exp1 LoCoMo benchmark
# ---------------------------------------------------------------------------
def print_table1(exp1_data):
    if exp1_data is None:
        print("[Table 1] exp1_dmem.json not found, skipping.")
        return
    print("\n" + "=" * 80)
    print("TABLE 1: LoCoMo Main Benchmark (D-MEM)")
    print("=" * 80)
    agg = exp1_data.get('aggregate', {})
    # Overall
    overall = agg.get('overall', {})
    f1 = overall.get('f1', {}).get('mean', 0)
    bleu1 = overall.get('bleu1', {}).get('mean', 0)
    bert_f1 = overall.get('bert_f1', {}).get('mean', 0)
    print(f"{'Metric':<20} {'F1':>8} {'BLEU-1':>8} {'BERT-F1':>8}")
    print(f"{'Overall':<20} {f1:>8.4f} {bleu1:>8.4f} {bert_f1:>8.4f}")

    for cat_key in sorted(agg.keys()):
        if cat_key.startswith('category_'):
            cat_metrics = agg[cat_key]
            cf1 = cat_metrics.get('f1', {}).get('mean', 0)
            cb1 = cat_metrics.get('bleu1', {}).get('mean', 0)
            cbf1 = cat_metrics.get('bert_f1', {}).get('mean', 0)
            print(f"{cat_key:<20} {cf1:>8.4f} {cb1:>8.4f} {cbf1:>8.4f}")


# ---------------------------------------------------------------------------
# Table 2 & Figures 1a/1b: Exp2 Scalability
# ---------------------------------------------------------------------------
def print_table2_and_figures(exp2_data, output_dir):
    if exp2_data is None:
        print("[Table 2] exp2_scalability.json not found, skipping.")
        return
    print("\n" + "=" * 80)
    print("TABLE 2: Scalability at T=200")
    print("=" * 80)
    print(f"{'Method':<15} {'API Calls':>10} {'Total Tokens':>13} {'Avg Lat(s)':>11} {'Store Size':>11}")

    summary = {}
    for key, logs in exp2_data.items():
        if key == 'metadata':
            continue
        if not logs:
            continue
        total_calls = sum(l.get('final_calls', 0) for l in logs)
        total_tokens = sum(l.get('final_tokens', {}).get('total_tokens', 0) for l in logs)
        store_sizes = [l.get('store_size', 0) for l in logs]
        # Average latency per turn across all samples
        all_lats = []
        for l in logs:
            for step in l.get('per_turn', []):
                all_lats.append(step.get('latency_sec', 0))
        avg_lat = np.mean(all_lats) if all_lats else 0
        avg_store = np.mean(store_sizes) if store_sizes else 0

        print(f"{key:<15} {total_calls:>10} {total_tokens:>13} {avg_lat:>11.3f} {avg_store:>11.1f}")
        summary[key] = {
            'total_calls': total_calls,
            'total_tokens': total_tokens,
            'avg_lat': avg_lat,
            'avg_store': avg_store,
            'logs': logs,
        }

    if not HAS_MPL:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Figure 1a: Per-Turn Write Latency
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, info in summary.items():
        # Average per-turn latency across samples
        max_t = max(len(l.get('per_turn', [])) for l in info['logs'])
        avg_by_turn = []
        for t in range(max_t):
            vals = []
            for l in info['logs']:
                steps = l.get('per_turn', [])
                if t < len(steps):
                    vals.append(steps[t]['latency_sec'])
            avg_by_turn.append(np.mean(vals) if vals else 0)
        ax.plot(range(1, len(avg_by_turn) + 1), avg_by_turn, label=key, alpha=0.8)
    ax.set_xlabel('Turn')
    ax.set_ylabel('Latency (sec)')
    ax.set_title('Figure 1a: Per-Turn Write Latency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig1a_latency.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved fig1a_latency.png")

    # Figure 1b: Cumulative Token Cost
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, info in summary.items():
        max_t = max(len(l.get('per_turn', [])) for l in info['logs'])
        avg_cum = []
        for t in range(max_t):
            vals = []
            for l in info['logs']:
                steps = l.get('per_turn', [])
                if t < len(steps):
                    vals.append(steps[t]['cumulative_tokens'])
            avg_cum.append(np.mean(vals) if vals else (avg_cum[-1] if avg_cum else 0))
        ax.plot(range(1, len(avg_cum) + 1), avg_cum, label=key, alpha=0.8)
    ax.set_xlabel('Turn')
    ax.set_ylabel('Cumulative Tokens')
    ax.set_title('Figure 1b: Cumulative Token Cost')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig1b_tokens.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved fig1b_tokens.png")


# ---------------------------------------------------------------------------
# Figure 2: Exp3 Noise Robustness
# ---------------------------------------------------------------------------
def print_figure2(exp3_data, output_dir):
    if exp3_data is None:
        print("[Figure 2] exp3_noise.json not found, skipping.")
        return
    print("\n" + "=" * 80)
    print("FIGURE 2 DATA: Noise Robustness")
    print("=" * 80)
    print(f"{'Key':<25} {'Noise%':>7} {'F1':>8} {'BERT-F1':>8} {'Store':>8}")

    method_curves = defaultdict(lambda: {'noise': [], 'f1': [], 'bert_f1': [], 'store': []})
    for key, info in exp3_data.items():
        if key == 'metadata':
            continue
        nr = info.get('noise_ratio', 0)
        method = info.get('method', key)
        agg = info.get('aggregate', {}).get('overall', {})
        f1 = agg.get('f1', {}).get('mean', 0)
        bf1 = agg.get('bert_f1', {}).get('mean', 0)
        avg_store = info.get('avg_store_size', 0)
        print(f"{key:<25} {nr*100:>6.0f}% {f1:>8.4f} {bf1:>8.4f} {avg_store:>8.1f}")
        method_curves[method]['noise'].append(nr * 100)
        method_curves[method]['f1'].append(f1)
        method_curves[method]['bert_f1'].append(bf1)
        method_curves[method]['store'].append(avg_store)

    if not HAS_MPL:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Figure 2: F1 vs Noise Ratio
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, data in method_curves.items():
        order = np.argsort(data['noise'])
        xs = np.array(data['noise'])[order]
        ys = np.array(data['f1'])[order]
        ax.plot(xs, ys, marker='o', label=method)
    ax.set_xlabel('Noise Ratio (%)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Figure 2: Noise Robustness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig2_noise.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved fig2_noise.png")

    # Figure 2b: Store Size vs Noise Ratio
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, data in method_curves.items():
        order = np.argsort(data['noise'])
        xs = np.array(data['noise'])[order]
        ys = np.array(data['store'])[order]
        ax.plot(xs, ys, marker='s', label=method)
    ax.set_xlabel('Noise Ratio (%)')
    ax.set_ylabel('Store Size')
    ax.set_title('Figure 2b: Store Size vs Noise Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig2b_store_size.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved fig2b_store_size.png")


# ---------------------------------------------------------------------------
# Table 3 & Figure 3: Exp4 Ablation
# ---------------------------------------------------------------------------
def print_table3_and_figure3(exp4_data, output_dir):
    if exp4_data is None:
        print("[Table 3] exp4_ablation.json not found, skipping.")
        return
    print("\n" + "=" * 80)
    print("TABLE 3: Ablation Study")
    print("=" * 80)
    print(f"{'Variant':<18} {'F1':>8} {'BERT-F1':>8} {'Tokens':>10} {'Store':>8}")

    # Collect for Pareto plot
    pareto_data = {}
    baseline_tokens = None

    for variant, info in exp4_data.items():
        if variant == 'metadata':
            continue
        agg = info.get('aggregate', {}).get('overall', {})
        f1 = agg.get('f1', {}).get('mean', 0)
        bf1 = agg.get('bert_f1', {}).get('mean', 0)
        tokens = info.get('total_tokens', 0)
        store = info.get('avg_store_size', 0)
        print(f"{variant:<18} {f1:>8.4f} {bf1:>8.4f} {tokens:>10} {store:>8.1f}")
        pareto_data[variant] = {'f1': f1, 'tokens': tokens}

        if variant == 'D-MEM' or baseline_tokens is None:
            pass
        # Use highest-token variant as baseline reference
        if baseline_tokens is None or tokens > baseline_tokens:
            baseline_tokens = tokens

    if not HAS_MPL or not pareto_data:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Figure 3: Pareto Frontier
    fig, ax = plt.subplots(figsize=(8, 6))
    max_tokens = max(d['tokens'] for d in pareto_data.values()) if pareto_data else 1
    for variant, d in pareto_data.items():
        savings = (1 - d['tokens'] / max(max_tokens, 1)) * 100
        ax.scatter(savings, d['f1'], s=100, zorder=5)
        ax.annotate(variant, (savings, d['f1']),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel('Token Savings (%)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Figure 3: Pareto Frontier (Ablation)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig3_pareto.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved fig3_pareto.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="D-MEM Result Analysis")
    parser.add_argument("--input_dir", type=str, default="results/dmem")
    parser.add_argument("--output_dir", type=str, default="results/dmem/figures")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    exp1 = load_json(os.path.join(input_dir, 'exp1_dmem.json'))
    exp2 = load_json(os.path.join(input_dir, 'exp2_scalability.json'))
    exp3 = load_json(os.path.join(input_dir, 'exp3_noise.json'))
    exp4 = load_json(os.path.join(input_dir, 'exp4_ablation.json'))

    print_table1(exp1)
    print_table2_and_figures(exp2, output_dir)
    print_figure2(exp3, output_dir)
    print_table3_and_figure3(exp4, output_dir)

    print("\n" + "=" * 80)
    print("Analysis complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
