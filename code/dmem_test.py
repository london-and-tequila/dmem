"""
D-MEM experiment driver.
Agents: DMemAgent, MemGPTAgent, advancedMemAgent (A-MEM baseline).
Experiments 1–4 as defined in the D-MEM plan.
"""

import os
import sys
import json
import time
import copy
import pickle
import logging
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from memory_layer import (
    LLMController, AgenticMemorySystem, token_tracker,
)
from llm_cache import init_global_cache
from dmem import DopamineMemorySystem
from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
from utils import calculate_metrics, aggregate_metrics
from noise_generator import inject_noise, get_noise_ratios
from test_advanced import advancedMemAgent

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class DMemAgent(advancedMemAgent):
    """Agent using DopamineMemorySystem with RPE gating."""

    def __init__(self, model, backend, retrieve_k, temperature_c5,
                 theta_low=0.3, theta_high=0.7, alpha=0.6, rpe_mode='rpe',
                 warmup_turns=10,
                 sglang_host="http://localhost", sglang_port=30000):
        # Do NOT call super().__init__() — create our own memory system
        self.memory_system = DopamineMemorySystem(
            theta_low=theta_low,
            theta_high=theta_high,
            alpha=alpha,
            rpe_mode=rpe_mode,
            warmup_turns=warmup_turns,
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retriever_llm = LLMController(
            backend=backend, model=model, api_key=None,
            sglang_host=sglang_host, sglang_port=sglang_port,
        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5

    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)


class MemGPTAgent:
    """Wrapper around pymemgpt (Letta) official library.

    Falls back to a thin shim if pymemgpt is not installed, so that experiment
    scripts can still be parsed / dry-run without the dependency.
    """

    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self._client = None
        self._agent_state = None
        self._memories_text = []
        try:
            from letta import create_client
            self._client = create_client()
            self._agent_state = self._client.create_agent(
                name=f"dmem_memgpt_{datetime.now().strftime('%H%M%S')}",
                llm_config={"model": model},
            )
        except Exception as e:
            print(f"[MemGPTAgent] Could not init letta/pymemgpt: {e}")
            print("[MemGPTAgent] Using fallback in-memory store.")

    def add_memory(self, content: str, time: str = None):
        if self._client and self._agent_state:
            try:
                self._client.send_message(
                    agent_id=self._agent_state.id,
                    message=content,
                    role="user",
                )
            except Exception as e:
                print(f"[MemGPTAgent] send_message error: {e}")
                self._memories_text.append(content)
        else:
            self._memories_text.append(content)

    def answer_question(self, question: str, category: int, answer: str):
        """Generate answer — mirrors advancedMemAgent interface."""
        if self._client and self._agent_state:
            try:
                resp = self._client.send_message(
                    agent_id=self._agent_state.id,
                    message=question,
                    role="user",
                )
                prediction = resp.messages[-1].get("text", str(resp.messages[-1]))
                return json.dumps({"answer": prediction}), question, ""
            except Exception as e:
                return json.dumps({"answer": ""}), question, ""
        # Fallback: trivial retrieval
        context = "\n".join(self._memories_text[-20:])
        return json.dumps({"answer": context[:200]}), question, context


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

DMEM_ABLATIONS = {
    'D-MEM':       {'alpha': 0.6, 'theta_low': 0.3, 'theta_high': 0.7, 'rpe_mode': 'rpe'},
    'D-MEM-rand':  {'alpha': 0.6, 'theta_low': 0.3, 'theta_high': 0.7, 'rpe_mode': 'random'},
    'D-MEM-high':  {'alpha': 0.6, 'theta_low': 0.3, 'theta_high': 0.9, 'rpe_mode': 'rpe'},
    'D-MEM-surp':  {'alpha': 1.0, 'theta_low': 0.3, 'theta_high': 0.7, 'rpe_mode': 'surprise_only'},
    'D-MEM-util':  {'alpha': 0.0, 'theta_low': 0.3, 'theta_high': 0.7, 'rpe_mode': 'utility_only'},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def flatten_turns(sample) -> List[dict]:
    """Flatten a LoCoMo sample's sessions into a list of {text, time, session_id}."""
    turns = []
    for sess_id, sess in sample.conversation.sessions.items():
        for turn in sess.turns:
            turns.append({
                'text': f"Speaker {turn.speaker} says : {turn.text}",
                'time': sess.date_time,
                'session_id': sess_id,
            })
    return turns


def create_agent(method: str, model: str, backend: str, retrieve_k: int,
                 temperature_c5: float, ablation_cfg: dict = None,
                 sglang_host: str = "http://localhost", sglang_port: int = 30000):
    """Factory to create the right agent for a given method name."""
    if method == 'A-MEM':
        return advancedMemAgent(model, backend, retrieve_k, temperature_c5,
                                sglang_host, sglang_port)
    elif method == 'MemGPT':
        return MemGPTAgent(model=model)
    elif method.startswith('D-MEM'):
        cfg = ablation_cfg or DMEM_ABLATIONS.get(method, DMEM_ABLATIONS['D-MEM'])
        return DMemAgent(model, backend, retrieve_k, temperature_c5,
                         theta_low=cfg['theta_low'],
                         theta_high=cfg['theta_high'],
                         alpha=cfg['alpha'],
                         rpe_mode=cfg['rpe_mode'],
                         sglang_host=sglang_host,
                         sglang_port=sglang_port)
    else:
        raise ValueError(f"Unknown method: {method}")


def evaluate_qa(agent, qa_list, logger):
    """Run QA evaluation on an agent that has already ingested memories."""
    results = []
    all_metrics = []
    all_categories = []
    for qa in qa_list:
        if int(qa.category) not in [1, 2, 3, 4, 5]:
            continue
        prediction, user_prompt, raw_ctx = agent.answer_question(
            qa.question, qa.category, qa.final_answer
        )
        try:
            prediction = json.loads(prediction)["answer"]
        except Exception:
            pass
        metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {}
        all_metrics.append(metrics)
        all_categories.append(qa.category)
        results.append({
            'question': qa.question,
            'prediction': prediction,
            'reference': qa.final_answer,
            'category': qa.category,
            'metrics': metrics,
        })
        logger.info(f"  Q: {qa.question[:80]}... | pred: {str(prediction)[:60]} | ref: {str(qa.final_answer)[:60]}")
    agg = aggregate_metrics(all_metrics, all_categories) if all_metrics else {}
    return results, agg


# ---------------------------------------------------------------------------
# Experiment 1: Standard LoCoMo Benchmark (parallelized)
# ---------------------------------------------------------------------------

def _run_exp1_sample(task):
    """Worker function for a single exp1 sample. Runs in a separate process."""
    sidx, sample, args_dict = task

    # Per-worker logger (console only — main process handles the file log)
    logger = setup_logger(f'exp1_w{sidx}')

    t0 = time.time()
    token_tracker.reset()

    agent = create_agent('D-MEM', args_dict['model'], args_dict['backend'],
                         args_dict['retrieve_k'], args_dict['temperature_c5'],
                         sglang_host=args_dict['sglang_host'],
                         sglang_port=args_dict['sglang_port'])

    # Ingest conversations
    turns = flatten_turns(sample)
    for t in turns:
        agent.add_memory(t['text'], time=t['time'])

    # QA
    res, _ = evaluate_qa(agent, sample.qa, logger)

    metrics_list = []
    cats_list = []
    for r in res:
        if r['metrics']:
            metrics_list.append(r['metrics'])
            cats_list.append(r['category'])

    routing = {}
    if hasattr(agent.memory_system, 'get_routing_summary'):
        routing = agent.memory_system.get_routing_summary()

    tokens = token_tracker.snapshot()
    elapsed = time.time() - t0

    return {
        'sample_idx': sidx,
        'results': res,
        'metrics': metrics_list,
        'categories': cats_list,
        'routing': routing,
        'tokens': tokens,
        'elapsed_sec': elapsed,
    }


def run_exp1(args):
    """Exp 1: D-MEM on standard LoCoMo10 (Table 1). Parallelized across samples."""
    from multiprocessing import Pool

    logger = setup_logger('exp1', os.path.join(args.log_dir, 'exp1.log'))
    logger.info("=== Experiment 1: LoCoMo Main Benchmark (parallel) ===")

    samples = load_locomo_dataset(args.dataset)
    if args.ratio < 1.0:
        samples = samples[:max(1, int(len(samples) * args.ratio))]

    # Serialisable args for worker processes
    args_dict = {
        'model': args.model,
        'backend': args.backend,
        'retrieve_k': args.retrieve_k,
        'temperature_c5': args.temperature_c5,
        'sglang_host': args.sglang_host,
        'sglang_port': args.sglang_port,
    }

    tasks = [(sidx, sample, args_dict) for sidx, sample in enumerate(samples)]
    NUM_WORKERS = min(10, len(samples))

    all_results = []
    all_metrics_list = []
    all_cats = []
    total_tokens = {
        'total_prompt_tokens': 0,
        'total_completion_tokens': 0,
        'total_tokens': 0,
        'call_count': 0,
    }

    logger.info(f"Running {len(samples)} samples with {NUM_WORKERS} parallel workers")

    with Pool(NUM_WORKERS) as pool:
        done = 0
        for output in pool.imap_unordered(_run_exp1_sample, tasks):
            done += 1
            sidx = output['sample_idx']

            all_results.extend(output['results'])
            all_metrics_list.extend(output['metrics'])
            all_cats.extend(output['categories'])

            # Aggregate token usage
            for k in ('total_prompt_tokens', 'total_completion_tokens',
                      'total_tokens', 'call_count'):
                total_tokens[k] += output['tokens'].get(k, 0)

            logger.info(
                f"[{done}/{len(samples)} done] Sample {sidx} finished in "
                f"{output['elapsed_sec']:.1f}s | tokens: {output['tokens']} | "
                f"routing: {output['routing']}"
            )

    overall_agg = aggregate_metrics(all_metrics_list, all_cats) if all_metrics_list else {}

    output = {
        'experiment': 'exp1',
        'method': 'D-MEM',
        'aggregate': overall_agg,
        'total_tokens': total_tokens,
        'results': all_results,
    }
    out_path = os.path.join(args.output_dir, 'exp1_dmem.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Exp1 results saved to {out_path}")
    logger.info(f"Aggregate metrics: {json.dumps(overall_agg, indent=2)}")
    logger.info(f"Total tokens: {total_tokens}")

    return output


# ---------------------------------------------------------------------------
# Experiment 2: Scalability & Cost
# ---------------------------------------------------------------------------

def _run_exp2_unit(task):
    """Worker: one (method, sample, group) pair for exp2. Runs in a separate process."""
    method, sidx, sample, args_dict, max_turns, noise_ratio, group_name = task

    token_tracker.reset()
    agent = create_agent(method, args_dict['model'], args_dict['backend'],
                         args_dict['retrieve_k'], args_dict['temperature_c5'],
                         sglang_host=args_dict['sglang_host'],
                         sglang_port=args_dict['sglang_port'])

    turns = flatten_turns(sample)
    turn_texts = [t['text'] for t in turns]
    turn_times = [t['time'] for t in turns]

    # Inject noise if needed
    if noise_ratio > 0:
        noisy_texts, _ = inject_noise(turn_texts, noise_ratio, seed=42 + sidx)
        while len(turn_times) < len(noisy_texts):
            turn_times.append(turn_times[-1] if turn_times else "")
        turn_texts = noisy_texts

    # Truncate
    turn_texts = turn_texts[:max_turns]
    turn_times = turn_times[:max_turns]

    t_start = time.time()
    sample_log = []
    for t_idx in range(len(turn_texts)):
        token_tracker.get_turn_token_cost()  # reset checkpoint
        t0 = time.time()
        agent.add_memory(turn_texts[t_idx],
                         time=turn_times[t_idx] if t_idx < len(turn_times) else None)
        latency = time.time() - t0
        turn_tokens = token_tracker.get_turn_token_cost()
        sample_log.append({
            'turn': t_idx + 1,
            'latency_sec': latency,
            'tokens': turn_tokens,
            'cumulative_tokens': token_tracker.total_tokens,
        })

    routing = {}
    if hasattr(agent, 'memory_system') and hasattr(agent.memory_system, 'get_routing_summary'):
        routing = agent.memory_system.get_routing_summary()

    return {
        'method': method,
        'group': group_name,
        'key': f"{method}_{group_name}",
        'sample_idx': sidx,
        'total_turns': len(turn_texts),
        'per_turn': sample_log,
        'final_tokens': token_tracker.snapshot(),
        'final_calls': token_tracker.call_count,
        'store_size': len(agent.memory_system.memories) if hasattr(agent, 'memory_system') else -1,
        'routing': routing,
        'elapsed_sec': time.time() - t_start,
    }


def run_exp2(args):
    """Exp 2: Turn-by-turn scalability (Figure 1a/1b + Table 2). Parallelized."""
    from multiprocessing import Pool

    logger = setup_logger('exp2', os.path.join(args.log_dir, 'exp2.log'))
    logger.info("=== Experiment 2: Scalability & Cost (parallel) ===")

    samples = load_locomo_dataset(args.dataset)
    if args.ratio < 1.0:
        samples = samples[:max(1, int(len(samples) * args.ratio))]

    max_turns = args.max_turns  # default 200
    methods = ['A-MEM', 'D-MEM']
    if args.include_memgpt:
        methods.append('MemGPT')

    groups = {'pure': 0.0}
    if args.include_noise_group:
        groups['noise_50'] = 0.5

    # Serialisable args for worker processes
    args_dict = {
        'model': args.model,
        'backend': args.backend,
        'retrieve_k': args.retrieve_k,
        'temperature_c5': args.temperature_c5,
        'sglang_host': args.sglang_host,
        'sglang_port': args.sglang_port,
    }

    # Build task list: all (method, sample, group) combinations
    tasks = []
    for group_name, noise_ratio in groups.items():
        for method in methods:
            for sidx, sample in enumerate(samples):
                tasks.append((method, sidx, sample, args_dict, max_turns,
                              noise_ratio, group_name))

    NUM_WORKERS = min(10, len(tasks))
    logger.info(f"Running {len(tasks)} tasks with {NUM_WORKERS} parallel workers")

    all_logs = defaultdict(list)

    with Pool(NUM_WORKERS) as pool:
        done = 0
        for output in pool.imap_unordered(_run_exp2_unit, tasks):
            done += 1
            key = output['key']
            all_logs[key].append({
                'sample_idx': output['sample_idx'],
                'total_turns': output['total_turns'],
                'per_turn': output['per_turn'],
                'final_tokens': output['final_tokens'],
                'final_calls': output['final_calls'],
                'store_size': output['store_size'],
                'routing': output.get('routing', {}),
            })
            logger.info(
                f"[{done}/{len(tasks)} done] {key} sample {output['sample_idx']} "
                f"finished in {output['elapsed_sec']:.1f}s | "
                f"turns: {output['total_turns']} | "
                f"tokens: {output['final_tokens']} | "
                f"routing: {output.get('routing', {})}"
            )

    # Convert defaultdict to regular dict for JSON serialization
    all_logs = dict(all_logs)

    out_path = os.path.join(args.output_dir, 'exp2_scalability.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_logs, f, indent=2)
    logger.info(f"Exp2 results saved to {out_path}")
    return all_logs


# ---------------------------------------------------------------------------
# Experiment 3: Noise Robustness
# ---------------------------------------------------------------------------

def _run_exp3_unit(task):
    """Worker: one (method, noise_ratio, sample) for exp3. Runs in a separate process."""
    method, sidx, sample, args_dict, noise_ratio, noise_key = task

    logger = setup_logger(f'exp3_w{noise_key}_{sidx}')
    t0 = time.time()
    token_tracker.reset()

    agent = create_agent(method, args_dict['model'], args_dict['backend'],
                         args_dict['retrieve_k'], args_dict['temperature_c5'],
                         sglang_host=args_dict['sglang_host'],
                         sglang_port=args_dict['sglang_port'])

    turns = flatten_turns(sample)
    turn_texts = [t['text'] for t in turns]
    turn_times = [t['time'] for t in turns]

    if noise_ratio > 0:
        noisy_texts, _ = inject_noise(turn_texts, noise_ratio, seed=42 + sidx)
        while len(turn_times) < len(noisy_texts):
            turn_times.append(turn_times[-1] if turn_times else "")
        turn_texts = noisy_texts

    for i, txt in enumerate(turn_texts):
        agent.add_memory(txt, time=turn_times[i] if i < len(turn_times) else None)

    # Evaluate on original QAs
    res, _ = evaluate_qa(agent, sample.qa, logger)
    metrics_list = [r['metrics'] for r in res if r['metrics']]
    cats_list = [r['category'] for r in res if r['metrics']]

    store_size = len(agent.memory_system.memories) if hasattr(agent, 'memory_system') else -1
    routing = {}
    if hasattr(agent, 'memory_system') and hasattr(agent.memory_system, 'get_routing_summary'):
        routing = agent.memory_system.get_routing_summary()

    return {
        'noise_key': noise_key,
        'method': method,
        'noise_ratio': noise_ratio,
        'sample_idx': sidx,
        'results': res,
        'metrics': metrics_list,
        'categories': cats_list,
        'store_size': store_size,
        'routing': routing,
        'tokens': token_tracker.snapshot(),
        'elapsed_sec': time.time() - t0,
    }


def run_exp3(args):
    """Exp 3: Noise robustness (Figure 2). Parallelized across (method, noise_ratio, sample)."""
    from multiprocessing import Pool

    logger = setup_logger('exp3', os.path.join(args.log_dir, 'exp3.log'))
    logger.info("=== Experiment 3: Noise Robustness (parallel) ===")

    samples = load_locomo_dataset(args.dataset)
    if args.ratio < 1.0:
        samples = samples[:max(1, int(len(samples) * args.ratio))]

    noise_ratios = get_noise_ratios()
    methods = ['A-MEM', 'D-MEM']
    if args.include_memgpt:
        methods.append('MemGPT')

    # Serialisable args for worker processes
    args_dict = {
        'model': args.model,
        'backend': args.backend,
        'retrieve_k': args.retrieve_k,
        'temperature_c5': args.temperature_c5,
        'sglang_host': args.sglang_host,
        'sglang_port': args.sglang_port,
    }

    # Build task list: all (method, noise_ratio, sample) combinations
    tasks = []
    for noise_ratio in noise_ratios:
        for method in methods:
            noise_key = f"{method}_noise{int(noise_ratio*100)}"
            for sidx, sample in enumerate(samples):
                tasks.append((method, sidx, sample, args_dict, noise_ratio, noise_key))

    NUM_WORKERS = min(10, len(tasks))
    logger.info(f"Running {len(tasks)} tasks with {NUM_WORKERS} parallel workers")

    # Collect results grouped by noise_key
    grouped = defaultdict(lambda: {
        'results': [], 'metrics': [], 'categories': [],
        'store_sizes': [], 'noise_ratio': 0.0, 'method': '',
    })

    with Pool(NUM_WORKERS) as pool:
        done = 0
        for output in pool.imap_unordered(_run_exp3_unit, tasks):
            done += 1
            nk = output['noise_key']
            grouped[nk]['results'].extend(output['results'])
            grouped[nk]['metrics'].extend(output['metrics'])
            grouped[nk]['categories'].extend(output['categories'])
            grouped[nk]['store_sizes'].append(output['store_size'])
            grouped[nk]['noise_ratio'] = output['noise_ratio']
            grouped[nk]['method'] = output['method']

            logger.info(
                f"[{done}/{len(tasks)} done] {nk} sample {output['sample_idx']} "
                f"finished in {output['elapsed_sec']:.1f}s | "
                f"store_size: {output['store_size']} | "
                f"routing: {output['routing']}"
            )

    # Aggregate per noise_key
    all_results = {}
    for nk, g in grouped.items():
        overall_agg = aggregate_metrics(g['metrics'], g['categories']) if g['metrics'] else {}
        all_results[nk] = {
            'noise_ratio': g['noise_ratio'],
            'method': g['method'],
            'aggregate': overall_agg,
            'store_sizes': g['store_sizes'],
            'avg_store_size': float(np.mean(g['store_sizes'])) if g['store_sizes'] else 0,
            'results': g['results'],
        }

    out_path = os.path.join(args.output_dir, 'exp3_noise.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Exp3 results saved to {out_path}")
    return all_results


# ---------------------------------------------------------------------------
# Experiment 4: Ablation
# ---------------------------------------------------------------------------

def run_exp4(args):
    """Exp 4: Ablation study (Table 3 + Figure 3)."""
    logger = setup_logger('exp4', os.path.join(args.log_dir, 'exp4.log'))
    logger.info("=== Experiment 4: Ablation ===")

    samples = load_locomo_dataset(args.dataset)
    if args.ratio < 1.0:
        samples = samples[:max(1, int(len(samples) * args.ratio))]

    all_results = {}

    for variant_name, cfg in DMEM_ABLATIONS.items():
        logger.info(f"--- {variant_name}: {cfg} ---")
        variant_metrics = []
        variant_cats = []
        variant_qas = []
        total_tokens_all = 0
        store_sizes = []

        for sidx, sample in enumerate(tqdm(samples, desc=variant_name)):
            token_tracker.reset()
            agent = create_agent(variant_name, args.model, args.backend,
                                 args.retrieve_k, args.temperature_c5,
                                 ablation_cfg=cfg,
                                 sglang_host=args.sglang_host,
                                 sglang_port=args.sglang_port)

            turns = flatten_turns(sample)
            for t in turns:
                agent.add_memory(t['text'], time=t['time'])

            res, agg = evaluate_qa(agent, sample.qa, logger)
            variant_qas.extend(res)
            for r in res:
                if r['metrics']:
                    variant_metrics.append(r['metrics'])
                    variant_cats.append(r['category'])

            total_tokens_all += token_tracker.total_tokens
            store_sizes.append(len(agent.memory_system.memories))

            if hasattr(agent.memory_system, 'get_routing_summary'):
                logger.info(f"  {variant_name} sample {sidx} routing: "
                            f"{agent.memory_system.get_routing_summary()}")

        overall_agg = aggregate_metrics(variant_metrics, variant_cats) if variant_metrics else {}
        all_results[variant_name] = {
            'config': cfg,
            'aggregate': overall_agg,
            'total_tokens': total_tokens_all,
            'avg_store_size': float(np.mean(store_sizes)) if store_sizes else 0,
            'results': variant_qas,
        }

    out_path = os.path.join(args.output_dir, 'exp4_ablation.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Exp4 results saved to {out_path}")
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="D-MEM Experiments")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["exp1", "exp2", "exp3", "exp4", "all"],
                        help="Which experiment to run")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--backend", type=str, default="openai")
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--retrieve_k", type=int, default=10)
    parser.add_argument("--temperature_c5", type=float, default=0.5)
    parser.add_argument("--max_turns", type=int, default=200,
                        help="Max turns for Exp2 scalability")
    parser.add_argument("--include_memgpt", action="store_true",
                        help="Include MemGPT baseline in Exp2/3")
    parser.add_argument("--include_noise_group", action="store_true",
                        help="Include 50%% noise group in Exp2")
    parser.add_argument("--output_dir", type=str, default="results/dmem")
    parser.add_argument("--log_dir", type=str, default="logs/dmem")
    parser.add_argument("--sglang_host", type=str, default="http://localhost")
    parser.add_argument("--sglang_port", type=int, default=30000)
    parser.add_argument("--cache_db", type=str, default=None,
                        help="Path to LLM cache SQLite DB (default: code/.llm_cache.db)")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable LLM response caching")
    args = parser.parse_args()

    # Convert relative paths
    args.dataset = os.path.join(os.path.dirname(__file__), args.dataset)
    args.output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    args.log_dir = os.path.join(os.path.dirname(__file__), args.log_dir)

    # Initialize LLM cache
    cache_path = args.cache_db or os.path.join(os.path.dirname(__file__), ".llm_cache.db")
    init_global_cache(db_path=cache_path, disabled=args.no_cache)
    if not args.no_cache:
        print(f"[LLM Cache] enabled — {cache_path}")

    if args.experiment == "exp1":
        run_exp1(args)
    elif args.experiment == "exp2":
        run_exp2(args)
    elif args.experiment == "exp3":
        run_exp3(args)
    elif args.experiment == "exp4":
        run_exp4(args)
    elif args.experiment == "all":
        run_exp1(args)
        run_exp2(args)
        run_exp3(args)
        run_exp4(args)


if __name__ == "__main__":
    main()
