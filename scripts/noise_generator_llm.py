"""
LLM-based noise generator for LoCoMo datasets.

Reads raw locomo10.json, injects LLM-generated contextual noise turns
(filler, status updates, tangents) at configurable ratios.
Output preserves LoCoMo JSON schema.
"""

import sys
import os
import json
import copy
import random
import uuid
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
from llm_cache import init_global_cache
from memory_layer import LLMController

# ---------------------------------------------------------------------------
# Noise prompt templates
# ---------------------------------------------------------------------------

NOISE_PROMPTS = {
    'filler': """You are simulating a real conversation. Given the recent dialogue:
{context}
Generate a brief filler/backchannel response from {speaker}. Examples: "hmm let me think", "oh interesting", "yeah that makes sense", "haha true". Keep it under 10 words. Return ONLY the filler text, nothing else.""",

    'status': """You are simulating a real conversation. Given the recent dialogue:
{context}
Generate a brief transient status update from {speaker} that would naturally interrupt. Examples: "brb getting coffee", "hold on my phone is ringing", "one sec let me check something". Keep it under 15 words. Return ONLY the status text.""",

    'tangent': """You are simulating a real conversation. Given the recent dialogue:
{context}
Generate a brief off-topic tangent from {speaker} (1 sentence, e.g., commenting on weather, food, a random thought). It should feel like a natural momentary distraction. Keep it under 20 words. Return ONLY the tangent text.""",
}

# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_noise_turn(speaker_a, speaker_b, recent_turns, noise_type, llm) -> dict:
    """Generate a single noise turn using the LLM."""
    speaker = random.choice([speaker_a, speaker_b])
    context = "\n".join(recent_turns[-3:]) if recent_turns else "(conversation just started)"
    prompt = NOISE_PROMPTS[noise_type].format(context=context, speaker=speaker)
    text = llm.get_completion(prompt).strip().strip('"')
    noise_id = f"NOISE:{uuid.uuid4().hex[:6]}"
    return {"speaker": speaker, "dia_id": noise_id, "text": text}


def inject_noise_into_dataset(raw_data, noise_ratio, llm, seed=42):
    """Inject LLM-generated noise turns into a LoCoMo dataset.

    Only modifies session turn lists; QA, event_summary, observation,
    and session_summary are preserved unchanged.
    """
    rng = random.Random(seed)
    noisy_data = copy.deepcopy(raw_data)

    for sample in noisy_data:
        conv = sample['conversation']
        speaker_a = conv.get('speaker_a', 'A')
        speaker_b = conv.get('speaker_b', 'B')

        for key in sorted(conv.keys()):
            if not key.startswith('session_') or not isinstance(conv[key], list):
                continue

            turns = conv[key]
            if not turns:
                continue

            num_noise = int(len(turns) * noise_ratio)
            if num_noise == 0:
                continue

            # Choose insertion positions (after the first turn)
            max_pos = len(turns)
            if max_pos < 2:
                continue
            positions = sorted(
                rng.sample(range(1, max_pos), min(num_noise, max_pos - 1))
            )

            offset = 0
            for pos in positions:
                insert_at = pos + offset
                recent = [t['text'] for t in turns[max(0, insert_at - 3):insert_at]]
                noise_type = rng.choices(
                    ['filler', 'status', 'tangent'],
                    weights=[0.4, 0.3, 0.3],
                )[0]
                noise_turn = generate_noise_turn(
                    speaker_a, speaker_b, recent, noise_type, llm
                )
                turns.insert(insert_at, noise_turn)
                offset += 1

    return noisy_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM-based noise generator for LoCoMo datasets"
    )
    parser.add_argument("--input", type=str, default="code/data/locomo10.json",
                        help="Path to raw LoCoMo JSON")
    parser.add_argument("--output_dir", type=str, default="data/locomo_noise/",
                        help="Output directory for noisy datasets")
    parser.add_argument("--ratios", type=int, nargs="+", default=[0, 25, 50, 75],
                        help="Noise ratios as percentages (e.g., 0 25 50 75)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--backend", type=str, default="openai")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_db", type=str, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--sglang_host", type=str, default="http://localhost")
    parser.add_argument("--sglang_port", type=int, default=30000)
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = os.path.join(os.path.dirname(__file__), '..')
    input_path = os.path.join(project_root, args.input)
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Init cache
    cache_path = args.cache_db or os.path.join(project_root, 'code', '.llm_cache.db')
    init_global_cache(db_path=cache_path, disabled=args.no_cache)
    if not args.no_cache:
        print(f"[LLM Cache] enabled — {cache_path}")

    # Init LLM
    llm_ctrl = LLMController(
        backend=args.backend, model=args.model,
        api_key=args.api_key, api_base=args.api_base,
        sglang_host=args.sglang_host, sglang_port=args.sglang_port,
    )
    llm = llm_ctrl.llm

    # Load raw dataset
    print(f"Loading raw dataset from {input_path}")
    with open(input_path, 'r') as f:
        raw_data = json.load(f)
    print(f"  Loaded {len(raw_data)} samples")

    # Count original turns
    def count_turns(data):
        total = 0
        for sample in data:
            conv = sample['conversation']
            for key in conv:
                if key.startswith('session_') and isinstance(conv[key], list):
                    total += len(conv[key])
        return total

    orig_turns = count_turns(raw_data)
    print(f"  Original turn count: {orig_turns}")

    # Generate for each ratio
    for ratio_pct in args.ratios:
        ratio = ratio_pct / 100.0
        print(f"\n--- Generating noise ratio {ratio_pct}% ---")

        if ratio_pct == 0:
            noisy_data = copy.deepcopy(raw_data)
        else:
            noisy_data = inject_noise_into_dataset(
                raw_data, ratio, llm, seed=args.seed
            )

        noisy_turns = count_turns(noisy_data)
        output_path = os.path.join(output_dir, f"locomo10_noise{ratio_pct}.json")
        with open(output_path, 'w') as f:
            json.dump(noisy_data, f, indent=2)
        print(f"  Saved to {output_path}")
        print(f"  Turn count: {orig_turns} -> {noisy_turns} (+{noisy_turns - orig_turns})")

    # Verification: try loading each output with load_locomo_dataset
    print("\n--- Verification ---")
    sys.path.insert(0, os.path.join(project_root, 'code'))
    from load_dataset import load_locomo_dataset
    for ratio_pct in args.ratios:
        path = os.path.join(output_dir, f"locomo10_noise{ratio_pct}.json")
        try:
            samples = load_locomo_dataset(path)
            print(f"  noise{ratio_pct}: OK ({len(samples)} samples loaded)")
        except Exception as e:
            print(f"  noise{ratio_pct}: FAILED — {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
