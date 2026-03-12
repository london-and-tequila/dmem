"""
Memory graph topology analysis and visualization.

Loads exported graph JSONs from run_eval.py, computes topology metrics,
generates comparison charts (node count, avg degree, density, routing funnel),
and optionally renders interactive PyVis HTML graphs.
"""

import sys
import os
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    nx = None
    HAS_NX = False

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False

# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_graph(path: str):
    """Load a graph JSON exported by export_graph_json() into NetworkX."""
    with open(path) as f:
        data = json.load(f)
    G = nx.DiGraph()
    for node in data.get('nodes', []):
        G.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
    for edge in data.get('links', []):
        G.add_edge(edge['source'], edge['target'])
    G.graph.update(data.get('graph', {}))
    return G, data


def parse_filename(filename: str) -> dict:
    """Parse model/method/noise/sample from graph filename.

    Expected format: {model_tag}_{method}_noise{N}_s{S}.json
    """
    stem = Path(filename).stem
    # Try pattern: {model}_{method}_noise{N}_s{S}
    m = re.match(r'^(.+?)_(d-mem|a-mem)_noise(\d+)_s(\d+)$', stem)
    if m:
        return {
            'model': m.group(1),
            'method': m.group(2),
            'noise': int(m.group(3)),
            'sample': int(m.group(4)),
            'filename': filename,
        }
    return None


# ---------------------------------------------------------------------------
# Topology metrics
# ---------------------------------------------------------------------------

def compute_metrics(G) -> dict:
    """Compute topology metrics for a single graph."""
    n = G.number_of_nodes()
    e = G.number_of_edges()
    degrees = dict(G.degree())
    return {
        'node_count': n,
        'edge_count': e,
        'avg_degree': sum(degrees.values()) / max(n, 1),
        'max_degree': max(degrees.values()) if degrees else 0,
        'density': nx.density(G),
        'weakly_connected_components': nx.number_weakly_connected_components(G),
        'avg_clustering': nx.average_clustering(G.to_undirected()) if n > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Comparison matrix
# ---------------------------------------------------------------------------

def build_comparison_df(graph_dir: str) -> pd.DataFrame:
    """Scan graph_dir, compute metrics for each graph, return DataFrame."""
    rows = []
    for fname in sorted(os.listdir(graph_dir)):
        if not fname.endswith('.json'):
            continue
        info = parse_filename(fname)
        if info is None:
            continue
        path = os.path.join(graph_dir, fname)
        try:
            G, raw = load_graph(path)
        except Exception as e:
            print(f"  Warning: failed to load {fname}: {e}")
            continue
        metrics = compute_metrics(G)
        row = {**info, **metrics}
        # Add routing info if available
        routing = raw.get('graph', {}).get('routing', {})
        if routing:
            row['skip_pct'] = routing.get('skip_pct', 0)
            row['construct_pct'] = routing.get('construct_pct', 0)
            row['evolve_pct'] = routing.get('evolve_pct', 0)
        rows.append(row)

    if not rows:
        print("No valid graph files found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def aggregate_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """Average metrics across samples for each (model, method, noise) config."""
    if df.empty:
        return df
    group_cols = ['model', 'method', 'noise']
    metric_cols = ['node_count', 'edge_count', 'avg_degree', 'max_degree',
                   'density', 'weakly_connected_components', 'avg_clustering']
    # Include routing columns if present
    for col in ['skip_pct', 'construct_pct', 'evolve_pct']:
        if col in df.columns:
            metric_cols.append(col)
    agg = df.groupby(group_cols)[metric_cols].mean().reset_index()
    return agg


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_metric_vs_noise(agg_df, metric, output_dir, ylabel=None):
    """Line plot: metric vs noise_ratio, one line per (model, method)."""
    if agg_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for (model, method), group in agg_df.groupby(['model', 'method']):
        group_sorted = group.sort_values('noise')
        label = f"{model} / {method}"
        linestyle = '--' if method == 'a-mem' else '-'
        ax.plot(group_sorted['noise'], group_sorted[metric],
                marker='o', linestyle=linestyle, label=label)
    ax.set_xlabel('Noise Ratio (%)')
    ax.set_ylabel(ylabel or metric)
    ax.set_title(f'{ylabel or metric} vs Noise Ratio')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{metric}_vs_noise.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_routing_funnel(agg_df, output_dir):
    """Stacked bar chart: SKIP / CONSTRUCT / EVOLVE proportions by noise."""
    if agg_df.empty or 'skip_pct' not in agg_df.columns:
        print("  Skipping routing funnel (no routing data)")
        return

    dmem_df = agg_df[agg_df['method'] == 'd-mem'].copy()
    if dmem_df.empty:
        return

    # Group by (model, noise) — one bar per config
    fig, ax = plt.subplots(figsize=(10, 6))
    dmem_sorted = dmem_df.sort_values(['model', 'noise'])
    labels = [f"{r['model']}\nn{int(r['noise'])}" for _, r in dmem_sorted.iterrows()]
    x = np.arange(len(labels))
    width = 0.6

    skip = dmem_sorted['skip_pct'].values
    construct = dmem_sorted['construct_pct'].values
    evolve = dmem_sorted['evolve_pct'].values

    ax.bar(x, skip, width, label='SKIP', color='#95a5a6')
    ax.bar(x, construct, width, bottom=skip, label='CONSTRUCT', color='#3498db')
    ax.bar(x, evolve, width, bottom=skip + construct, label='EVOLVE', color='#e74c3c')

    ax.set_ylabel('Percentage (%)')
    ax.set_title('D-MEM Routing Funnel by Model & Noise')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'routing_funnel.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_density_heatmap(agg_df, output_dir):
    """Heatmap: density by model x noise, for each method."""
    if agg_df.empty:
        return
    for method, mdf in agg_df.groupby('method'):
        pivot = mdf.pivot_table(values='density', index='model', columns='noise')
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, max(3, len(pivot) * 0.8)))
        im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{int(c)}%" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel('Noise Ratio')
        ax.set_title(f'Graph Density — {method}')
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.values[i, j]:.4f}",
                        ha='center', va='center', fontsize=8)
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f'density_heatmap_{method}.png')
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# PyVis interactive graph
# ---------------------------------------------------------------------------

def render_interactive(G, raw_data, output_html, title="Memory Graph"):
    """Render an interactive HTML visualization of the memory graph."""
    if not HAS_PYVIS:
        print("  pyvis not installed, skipping interactive render")
        return

    net = Network(height="800px", width="100%", directed=True,
                  heading=title, bgcolor="#222222", font_color="white")
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=100)

    for node_id, attrs in G.nodes(data=True):
        content = attrs.get('content', '')[:40]
        keywords = attrs.get('keywords', [])
        importance = attrs.get('importance_score', 1.0)
        # Size by importance
        size = max(8, min(30, importance * 15))
        color = '#3498db'
        net.add_node(str(node_id), label=content, color=color,
                     size=size, title=f"Keywords: {keywords}")

    for u, v in G.edges():
        net.add_edge(str(u), str(v), color='#7f8c8d')

    net.save_graph(output_html)
    print(f"  Saved interactive graph: {output_html}")


# ---------------------------------------------------------------------------
# Result summary from eval JSONs
# ---------------------------------------------------------------------------

def load_eval_results(result_dir: str) -> pd.DataFrame:
    """Load eval result JSONs and extract key metrics."""
    rows = []
    for fname in sorted(os.listdir(result_dir)):
        if not fname.startswith('eval_') or not fname.endswith('.json'):
            continue
        # Skip graph subdirectory
        path = os.path.join(result_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        overall = data.get('aggregate', {}).get('overall', {})
        tokens = data.get('total_tokens', {})
        rows.append({
            'file': fname,
            'method': data.get('method', ''),
            'model': data.get('model', ''),
            'noise_ratio': data.get('noise_ratio', 0),
            'f1': overall.get('f1', {}).get('mean', 0),
            'bert_f1': overall.get('bert_f1', {}).get('mean', 0),
            'bleu1': overall.get('bleu1', {}).get('mean', 0),
            'total_tokens': tokens.get('total_tokens', 0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Memory graph topology analysis")
    parser.add_argument("--graph_dir", type=str, default="results/dmem/graphs/")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Alias for --graph_dir (backwards compat)")
    parser.add_argument("--result_dir", type=str, default="results/dmem/")
    parser.add_argument("--output_dir", type=str, default="results/dmem/graph_analysis/")
    parser.add_argument("--render_html", action="store_true",
                        help="Generate interactive PyVis HTML graphs")
    args = parser.parse_args()

    # Handle --results_dir alias
    if args.results_dir:
        args.graph_dir = args.results_dir

    # Resolve paths
    project_root = os.path.join(os.path.dirname(__file__), '..')
    graph_dir = os.path.join(project_root, args.graph_dir)
    result_dir = os.path.join(project_root, args.result_dir)
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not HAS_NX:
        print("ERROR: networkx is required. pip install networkx")
        sys.exit(1)

    # --- Graph topology analysis ---
    print("=== Graph Topology Analysis ===")
    print(f"Scanning: {graph_dir}")

    if not os.path.isdir(graph_dir):
        print(f"Graph directory not found: {graph_dir}")
        print("Run evaluations with --export_graph first.")
        sys.exit(1)

    df = build_comparison_df(graph_dir)
    if df.empty:
        print("No graph data found. Exiting.")
        sys.exit(0)

    print(f"\nFound {len(df)} graph files")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Noise levels: {sorted(df['noise'].unique())}")

    # Aggregate by (model, method, noise)
    agg_df = aggregate_by_config(df)

    # Save comparison table
    table_path = os.path.join(output_dir, 'topology_comparison.csv')
    agg_df.to_csv(table_path, index=False, float_format='%.4f')
    print(f"\nComparison table saved: {table_path}")
    print(agg_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Plots ---
    print("\n=== Generating Plots ===")
    plot_metric_vs_noise(agg_df, 'node_count', output_dir, 'Node Count')
    plot_metric_vs_noise(agg_df, 'avg_degree', output_dir, 'Average Degree')
    plot_metric_vs_noise(agg_df, 'density', output_dir, 'Graph Density')
    plot_metric_vs_noise(agg_df, 'avg_clustering', output_dir, 'Avg Clustering Coefficient')
    plot_routing_funnel(agg_df, output_dir)
    plot_density_heatmap(agg_df, output_dir)

    # --- Interactive HTML graphs ---
    if args.render_html:
        print("\n=== Rendering Interactive Graphs ===")
        if not HAS_PYVIS:
            print("pyvis not installed. pip install pyvis")
        else:
            # Render one graph per (method, noise) at the highest noise level
            max_noise = df['noise'].max()
            for method in df['method'].unique():
                subset = df[(df['method'] == method) & (df['noise'] == max_noise)]
                if subset.empty:
                    continue
                # Pick first sample of first model
                row = subset.iloc[0]
                fpath = os.path.join(graph_dir, row['filename'])
                G, raw = load_graph(fpath)
                html_path = os.path.join(
                    output_dir, f'graph_{method}_noise{int(max_noise)}.html'
                )
                render_interactive(
                    G, raw, html_path,
                    title=f"{method} — noise {int(max_noise)}% ({row['model']})"
                )

    # --- Eval result summary ---
    print("\n=== Eval Result Summary ===")
    eval_df = load_eval_results(result_dir)
    if not eval_df.empty:
        summary_path = os.path.join(output_dir, 'eval_summary.csv')
        eval_df.to_csv(summary_path, index=False, float_format='%.4f')
        print(f"Eval summary saved: {summary_path}")
        print(eval_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("No eval result files found in", result_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
