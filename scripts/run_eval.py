"""
Unified evaluation entry point for D-MEM / A-MEM across models and noise levels.

Each run = one (model, method, noise_ratio) combination.
W&B `group` parameter aggregates same-noise runs for cross-model comparison.
"""

import sys
import os
import json
import time
import argparse
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False

from dmem_test import (
    create_agent, evaluate_qa, flatten_turns,
    save_experiment_result, setup_logger, build_metadata,
    _wandb_enabled, _wandb_finish_and_upload,
)
from memory_layer import token_tracker
from llm_cache import init_global_cache
from load_dataset import load_locomo_dataset
from utils import aggregate_metrics

# ---------------------------------------------------------------------------
# W&B init with group support
# ---------------------------------------------------------------------------

def wandb_init_grouped(args):
    if not HAS_WANDB or not getattr(args, 'wandb', False):
        return
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:
        git_hash = "unknown"

    group = args.wandb_group or f"noise_{args.noise_ratio}"
    model_tag = args.model.replace('/', '-')
    ts = datetime.now().strftime('%m%d_%H%M')
    suffix = f"_{args.router_mode}" if args.router_mode else ""
    tags = [args.method, args.model, f"noise{args.noise_ratio}", args.backend]
    if args.router_mode:
        tags.append(args.router_mode)
    wandb.init(
        project="d-mem-eval",
        group=group,
        name=f"{args.method}_{model_tag}_n{args.noise_ratio}{suffix}_{ts}",
        config={**vars(args), 'git_hash': git_hash},
        tags=tags,
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_single_eval(args):
    start_time = datetime.now()
    model_tag = args.model.replace('/', '-')
    log_name = f"eval_{args.method}_{model_tag}_n{args.noise_ratio}"
    logger = setup_logger(log_name, os.path.join(args.log_dir, f'{log_name}.log'))

    logger.info(f"=== Eval: method={args.method} model={args.model} "
                f"noise={args.noise_ratio} backend={args.backend} ===")

    # Load dataset
    dataset_path = os.path.join(args.dataset_dir, f"locomo10_noise{args.noise_ratio}.json")
    if not os.path.exists(dataset_path):
        # Fall back to raw dataset for noise_ratio=0
        if args.noise_ratio == 0:
            dataset_path = os.path.join(
                os.path.dirname(__file__), '..', 'code', 'data', 'locomo10.json'
            )
        else:
            logger.error(f"Dataset not found: {dataset_path}")
            return None

    samples = load_locomo_dataset(dataset_path)
    if args.ratio < 1.0:
        samples = samples[:max(1, int(len(samples) * args.ratio))]

    logger.info(f"Loaded {len(samples)} samples from {dataset_path}")

    # W&B
    wandb_init_grouped(args)

    # Map method name
    method_name = 'D-MEM' if args.method == 'd-mem' else 'A-MEM'

    all_results, all_metrics_list, all_cats = [], [], []
    all_per_turn_logs, all_routing = [], []
    total_tokens = {
        'total_prompt_tokens': 0, 'total_completion_tokens': 0,
        'total_tokens': 0, 'call_count': 0,
    }

    for sidx, sample in enumerate(tqdm(samples, desc=log_name)):
        token_tracker.reset()

        ablation_cfg = None
        if args.router_mode and method_name == 'D-MEM':
            ablation_cfg = {
                'alpha': args.alpha,
                'theta_low': args.theta_low,
                'theta_high': args.theta_high,
                'rpe_mode': args.router_mode,
            }

        agent = create_agent(
            method_name, args.model, args.backend,
            args.retrieve_k, args.temperature_c5,
            ablation_cfg=ablation_cfg,
            sglang_host=args.sglang_host, sglang_port=args.sglang_port,
        )

        # Ingest turns
        turns = flatten_turns(sample)
        per_turn_log = []
        for t_idx, t in enumerate(turns):
            token_tracker.get_turn_token_cost()  # checkpoint
            t0 = time.time()
            agent.add_memory(t['text'], time=t['time'])
            latency_ms = (time.time() - t0) * 1000
            turn_tokens = token_tracker.get_turn_token_cost()
            per_turn_log.append({
                'turn': t_idx + 1,
                'latency_ms': latency_ms,
                'turn_tokens': turn_tokens,
                'cumulative_tokens': token_tracker.total_tokens,
            })
            if _wandb_enabled():
                wandb.log({
                    'turn_latency_ms': latency_ms,
                    'turn_tokens': turn_tokens,
                    'cumulative_tokens': token_tracker.total_tokens,
                })

        # QA evaluation
        res, _ = evaluate_qa(agent, sample.qa, logger)
        metrics_list = [r['metrics'] for r in res if r['metrics']]
        cats_list = [r['category'] for r in res if r['metrics']]
        all_results.extend(res)
        all_metrics_list.extend(metrics_list)
        all_cats.extend(cats_list)
        all_per_turn_logs.append(per_turn_log)

        # Token accumulation
        tokens = token_tracker.snapshot()
        for k in total_tokens:
            total_tokens[k] += tokens.get(k, 0)

        # Routing summary (D-MEM only)
        routing = {}
        if hasattr(agent.memory_system, 'get_routing_summary'):
            routing = agent.memory_system.get_routing_summary()
        all_routing.append(routing)

        # Graph export
        if args.export_graph and hasattr(agent.memory_system, 'export_graph_json'):
            graph_data = agent.memory_system.export_graph_json()
            graph_dir = os.path.join(args.output_dir, 'graphs')
            os.makedirs(graph_dir, exist_ok=True)
            graph_path = os.path.join(
                graph_dir,
                f'{model_tag}_{args.method}_noise{args.noise_ratio}_s{sidx}.json'
            )
            with open(graph_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
            logger.info(f"  Graph exported: {graph_path}")

        # W&B per-sample log
        if _wandb_enabled():
            sample_agg = aggregate_metrics(metrics_list, cats_list) if metrics_list else {}
            overall = sample_agg.get('overall', {})
            wandb.log({
                'sample_f1': overall.get('f1', {}).get('mean', 0),
                'sample_bert_f1': overall.get('bert_f1', {}).get('mean', 0),
                'sample_tokens': tokens.get('total_tokens', 0),
                'routing_skip_pct': routing.get('skip_pct', 0),
                'store_size': routing.get('store_size', 0),
            })

        logger.info(
            f"[{sidx+1}/{len(samples)}] tokens={tokens.get('total_tokens', 0)} "
            f"routing={routing}"
        )

    # Aggregate
    overall_agg = aggregate_metrics(all_metrics_list, all_cats) if all_metrics_list else {}

    output = {
        'experiment': f'eval_{args.method}_noise{args.noise_ratio}',
        'method': method_name,
        'model': args.model,
        'noise_ratio': args.noise_ratio,
        'aggregate': overall_agg,
        'total_tokens': total_tokens,
        'per_turn_logs': all_per_turn_logs,
        'routing_summaries': all_routing,
        'results': all_results,
    }

    # Save
    tag = f'eval_{args.method}_{model_tag}_noise{args.noise_ratio}'
    semantic_path = save_experiment_result(output, args, tag, start_time, datetime.now())

    # W&B finalize
    if _wandb_enabled():
        ov = overall_agg.get('overall', {})
        wandb.run.summary.update({
            'final_f1': ov.get('f1', {}).get('mean', 0),
            'final_bert_f1': ov.get('bert_f1', {}).get('mean', 0),
            'total_tokens': total_tokens['total_tokens'],
        })
    _wandb_finish_and_upload(semantic_path)

    logger.info(f"Results saved: {semantic_path}")
    logger.info(f"Aggregate: {json.dumps(overall_agg.get('overall', {}), indent=2)}")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="D-MEM Unified Evaluation")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--method", type=str, default="d-mem",
                        choices=["d-mem", "a-mem", "dmem", "amem"])
    parser.add_argument("--noise_ratio", type=int, default=0,
                        help="Noise percentage (0, 25, 50, 75)")
    parser.add_argument("--backend", type=str, default="openai",
                        choices=["openai", "litellm", "sglang", "ollama"])
    parser.add_argument("--dataset_dir", type=str, default="data/locomo_noise/")
    parser.add_argument("--output_dir", type=str, default="results/dmem/")
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Fraction of dataset to use (for quick tests)")
    parser.add_argument("--retrieve_k", type=int, default=10)
    parser.add_argument("--temperature_c5", type=float, default=0.5)
    parser.add_argument("--sglang_host", type=str, default="http://localhost")
    parser.add_argument("--sglang_port", type=int, default=30000)
    parser.add_argument("--cache_db", type=str, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="W&B group name (default: auto = noise_{ratio})")
    parser.add_argument("--noise", type=int, default=None,
                        help="Alias for --noise_ratio")
    parser.add_argument("--router_mode", type=str, default=None,
                        choices=["rpe", "rpe_v2", "rpe_v3", "random", "surprise_only", "utility_only"],
                        help="RPE router mode for D-MEM ablation")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="RPE alpha (surprise vs utility weight)")
    parser.add_argument("--theta_low", type=float, default=0.3,
                        help="RPE low threshold")
    parser.add_argument("--theta_high", type=float, default=0.7,
                        help="RPE high threshold")
    parser.add_argument("--export_graph", action="store_true", default=None,
                        help="Export memory graph JSON (default: True)")
    parser.add_argument("--no_export_graph", action="store_true",
                        help="Disable graph export")
    args = parser.parse_args()

    # Handle --method alias (normalize to hyphenated form)
    if args.method == "dmem":
        args.method = "d-mem"
    elif args.method == "amem":
        args.method = "a-mem"

    # Handle --noise alias
    if args.noise is not None:
        args.noise_ratio = args.noise

    # Default export_graph: True for all methods
    if args.no_export_graph:
        args.export_graph = False
    elif args.export_graph is None:
        args.export_graph = True

    # Resolve paths relative to project root
    project_root = os.path.join(os.path.dirname(__file__), '..')
    args.dataset_dir = os.path.join(project_root, args.dataset_dir)
    args.output_dir = os.path.join(project_root, args.output_dir)
    args.log_dir = os.path.join(project_root, args.log_dir)

    # Init cache
    cache_path = args.cache_db or os.path.join(project_root, 'code', '.llm_cache.db')
    init_global_cache(db_path=cache_path, disabled=args.no_cache)
    if not args.no_cache:
        print(f"[LLM Cache] enabled — {cache_path}")

    run_single_eval(args)


if __name__ == "__main__":
    main()
