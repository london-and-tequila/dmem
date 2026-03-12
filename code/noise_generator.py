"""
Noise generator for D-MEM Experiment 3: Noise Robustness.
Injects filler noise at configurable ratios into conversation turn lists.
"""

import random
from typing import List, Tuple

# Noise templates
FILLER_NOISE = [
    "hmm", "ok", "yeah", "lol", "I see", "nice", "right", "sure",
    "uh huh", "mm", "gotcha", "cool", "oh", "haha", "yep", "alright",
    "okay", "no worries", "fine", "interesting",
]

OFF_TOPIC_NOISE = [
    "The weather is really nice today.",
    "I wonder what's for lunch.",
    "Did you see that new movie?",
    "My phone battery is dying.",
    "I need to buy groceries later.",
    "Traffic was terrible this morning.",
    "I should go to the gym soon.",
    "What time is it?",
    "I forgot my umbrella.",
    "This coffee is really good.",
]

SPEAKERS = ["Speaker A", "Speaker B"]


def generate_noise_turn(real_turns: List[str], noise_types: List[str] = None) -> str:
    """Generate a single noise turn."""
    if noise_types is None:
        noise_types = ['filler', 'off_topic', 'repetition']

    noise_type = random.choice(noise_types)

    if noise_type == 'filler':
        speaker = random.choice(SPEAKERS)
        filler = random.choice(FILLER_NOISE)
        return f"{speaker} says : {filler}"

    elif noise_type == 'off_topic':
        speaker = random.choice(SPEAKERS)
        topic = random.choice(OFF_TOPIC_NOISE)
        return f"{speaker} says : {topic}"

    elif noise_type == 'repetition' and real_turns:
        # Repeat an earlier turn
        return random.choice(real_turns)

    # Fallback to filler
    speaker = random.choice(SPEAKERS)
    return f"{speaker} says : {random.choice(FILLER_NOISE)}"


def inject_noise(turns: List[str],
                 noise_ratio: float,
                 noise_types: List[str] = None,
                 seed: int = 42) -> Tuple[List[str], List[bool]]:
    """
    Inject noise into a list of conversation turns.

    Args:
        turns: Original conversation turns (strings).
        noise_ratio: Fraction of noise turns to add (0.0 to 1.0).
                     e.g., 0.5 means for every 2 real turns, add 1 noise turn.
        noise_types: List of noise types to use: 'filler', 'off_topic', 'repetition'.
        seed: Random seed for reproducibility.

    Returns:
        (noisy_turns, is_noise_mask):
            noisy_turns — mixed list of real + noise turns.
            is_noise_mask — bool list, True for noise turns.
    """
    rng = random.Random(seed)

    if noise_ratio <= 0:
        return list(turns), [False] * len(turns)

    num_noise = int(len(turns) * noise_ratio)
    noise_turns = []
    for _ in range(num_noise):
        noise_turns.append(generate_noise_turn(turns, noise_types))

    # Build combined list: insert noise at random positions
    combined = [(t, False) for t in turns]
    for nt in noise_turns:
        pos = rng.randint(0, len(combined))
        combined.insert(pos, (nt, True))

    noisy_turns = [t for t, _ in combined]
    is_noise = [n for _, n in combined]
    return noisy_turns, is_noise


def get_noise_ratios() -> List[float]:
    """Standard noise ratios for Experiment 3."""
    return [0.0, 0.2, 0.4, 0.6, 0.8]
