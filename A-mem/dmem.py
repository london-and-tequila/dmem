"""
D-MEM: Dopamine-inspired Memory Management for LLM Agents
CriticRouter + DopamineMemorySystem
"""

import json
import math
import time
import random
import numpy as np
from collections import deque
from typing import Optional, Tuple, List
from sklearn.metrics.pairwise import cosine_similarity

from memory_layer import (
    AgenticMemorySystem, MemoryNote, LLMController,
    SimpleEmbeddingRetriever, token_tracker,
)


class CriticRouter:
    """RPE-based routing: decides SKIP / CONSTRUCT_ONLY / FULL_EVOLUTION for each input."""

    def __init__(self,
                 retriever: SimpleEmbeddingRetriever,
                 llm_controller: LLMController,
                 theta_low: float = 0.3,
                 theta_high: float = 0.7,
                 alpha: float = 0.6,
                 mode: str = 'rpe',
                 warmup_turns: int = 10):
        self.retriever = retriever
        self.llm_controller = llm_controller
        self.theta_low = theta_low
        self.theta_high = theta_high
        self.alpha = alpha
        self.mode = mode  # 'rpe' | 'random' | 'surprise_only' | 'utility_only'
        self.warmup_turns = warmup_turns
        self._turn_count = 0

        # Sliding window for Z-score normalization of surprise
        self._sim_window: deque = deque(maxlen=50)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_rpe(self, content: str) -> Tuple[float, str]:
        """Return (rpe_score, tier) where tier ∈ {SKIP, CONSTRUCT_ONLY, FULL_EVOLUTION}."""
        self._turn_count += 1

        # Cold start: force CONSTRUCT_ONLY during warmup
        if self._turn_count <= self.warmup_turns:
            return 0.5, 'CONSTRUCT_ONLY'

        if self.mode == 'random':
            rpe = random.random()
        elif self.mode == 'surprise_only':
            rpe = self._compute_surprise(content)
        elif self.mode == 'utility_only':
            rpe = self._compute_utility(content)
        else:  # 'rpe' (default)
            surprise = self._compute_surprise(content)
            utility = self._compute_utility(content)
            rpe = self.alpha * surprise + (1 - self.alpha) * utility

        tier = self._route(rpe)
        return rpe, tier

    # ------------------------------------------------------------------
    # Surprise (embedding-based, 0 LLM cost)
    # ------------------------------------------------------------------
    def _compute_surprise(self, content: str) -> float:
        """Z-score normalized surprise to counter embedding anisotropy."""
        if (self.retriever.embeddings is None or
                len(self.retriever.corpus) < 5):
            return 1.0  # cold start override

        query_emb = self.retriever.model.encode([content])
        sims = cosine_similarity(query_emb, self.retriever.embeddings)[0]
        raw_sim = float(np.max(sims))

        self._sim_window.append(raw_sim)

        if len(self._sim_window) < 5:
            return 1.0  # not enough data for meaningful stats

        mu = float(np.mean(self._sim_window))
        sigma = float(np.std(self._sim_window))
        z = (mu - raw_sim) / max(sigma, 0.01)
        surprise = 1.0 / (1.0 + math.exp(-z))  # sigmoid → [0, 1]
        return surprise

    # ------------------------------------------------------------------
    # Utility (lightweight LLM call, ~50 tokens)
    # ------------------------------------------------------------------
    def _compute_utility(self, content: str) -> float:
        """Structured-output LLM call to rate long-term factual value (0-10)."""
        prompt = (
            "Rate 0-10 the long-term factual value of this input for a "
            f"personal assistant: '{content}'"
        )
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "utility",
                "schema": {
                    "type": "object",
                    "properties": {
                        "utility_score": {"type": "integer"}
                    },
                    "required": ["utility_score"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
        try:
            raw = self.llm_controller.llm.get_completion(
                prompt, response_format=response_format, temperature=0.0
            )
            score = json.loads(raw).get("utility_score", 5)
            return max(0.0, min(1.0, score / 10.0))
        except Exception:
            return 0.5  # neutral fallback

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def _route(self, rpe: float) -> str:
        if rpe < self.theta_low:
            return 'SKIP'
        elif rpe < self.theta_high:
            return 'CONSTRUCT_ONLY'
        else:
            return 'FULL_EVOLUTION'


class DopamineMemorySystem(AgenticMemorySystem):
    """Memory system with RPE-gated write path."""

    def __init__(self,
                 theta_low: float = 0.3,
                 theta_high: float = 0.7,
                 alpha: float = 0.6,
                 rpe_mode: str = 'rpe',
                 warmup_turns: int = 10,
                 # pass-through to AgenticMemorySystem
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000):
        super().__init__(
            model_name=model_name,
            llm_backend=llm_backend,
            llm_model=llm_model,
            evo_threshold=evo_threshold,
            api_key=api_key,
            api_base=api_base,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.router = CriticRouter(
            retriever=self.retriever,
            llm_controller=self.llm_controller,
            theta_low=theta_low,
            theta_high=theta_high,
            alpha=alpha,
            mode=rpe_mode,
            warmup_turns=warmup_turns,
        )
        self.stm_buffer: deque = deque(maxlen=50)
        self.routing_stats = {
            'skip': 0, 'construct': 0, 'evolve': 0,
            'per_step': [],
        }

    def add_note(self, content: str, time: str = None, **kwargs) -> Optional[str]:
        """RPE-gated add_note with 3 tiers."""
        t0_turn = time_module_time()
        tokens_before = token_tracker.total_tokens

        rpe, tier = self.router.compute_rpe(content)

        note_id = None

        if tier == 'SKIP':
            # Tier 1: store in STM only, 0 extra LLM calls
            self.stm_buffer.append(content)
            self.routing_stats['skip'] += 1

        elif tier == 'CONSTRUCT_ONLY':
            # Tier 2: analyze content (1 LLM call) but NO evolution
            note = MemoryNote(content=content, llm_controller=self.llm_controller,
                              timestamp=time, **kwargs)
            self.memories[note.id] = note
            doc = ("content:" + note.content + " context:" + note.context +
                   " keywords: " + ", ".join(note.keywords) +
                   " tags: " + ", ".join(note.tags))
            self.retriever.add_documents([doc])
            note_id = note.id
            self.routing_stats['construct'] += 1

        else:  # FULL_EVOLUTION
            # Tier 3: full A-MEM pipeline (construct + evolve)
            note_id = super().add_note(content, time=time, **kwargs)
            self.routing_stats['evolve'] += 1

        latency_ms = (time_module_time() - t0_turn) * 1000
        tokens_used = token_tracker.total_tokens - tokens_before

        self.routing_stats['per_step'].append({
            'turn': len(self.routing_stats['per_step']) + 1,
            'rpe': rpe,
            'tier': tier,
            'tokens': tokens_used,
            'latency_ms': latency_ms,
        })

        return note_id

    def get_routing_summary(self) -> dict:
        total = (self.routing_stats['skip'] +
                 self.routing_stats['construct'] +
                 self.routing_stats['evolve'])
        return {
            'total_turns': total,
            'skip': self.routing_stats['skip'],
            'construct': self.routing_stats['construct'],
            'evolve': self.routing_stats['evolve'],
            'skip_pct': self.routing_stats['skip'] / max(total, 1) * 100,
            'construct_pct': self.routing_stats['construct'] / max(total, 1) * 100,
            'evolve_pct': self.routing_stats['evolve'] / max(total, 1) * 100,
            'store_size': len(self.memories),
        }


# Alias so we don't shadow the builtin `time` parameter name
time_module_time = time.time
