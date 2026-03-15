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
    SimpleEmbeddingRetriever, token_tracker, simple_tokenize,
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
        self.mode = mode  # 'rpe' | 'rpe_v2' | 'rpe_v3' | 'random' | 'surprise_only' | 'utility_only'
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
        elif self.mode in ('rpe_v2', 'rpe_v3'):
            utility = self._compute_utility(content)
            # Hard cutoff: TRANSIENT (utility ≈ 0) → skip immediately, 0 surprise cost
            if utility < 0.05:
                return 0.0, 'SKIP'
            surprise = self._compute_surprise(content)
            beta = 0.4 if self.mode == 'rpe_v3' else 0.1
            rpe = min(1.0, utility * (surprise + beta))
        else:  # 'rpe' (original linear)
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
        """Lightweight LLM call: lifecycle classification → score (no entity extraction)."""
        prompt = (
            "Classify this utterance's long-term memory value for an AI assistant.\n\n"
            f"Utterance: '{content}'\n\n"
            "Lifecycle:\n"
            "  TRANSIENT — pure filler with zero informational content "
            "(greetings, 'ok', 'lol', 'brb', 'thanks', backchannel, weather small-talk)\n"
            "  SHORT_TERM — days-to-weeks relevance (plans, temporary tasks, daily activities)\n"
            "  PERSISTENT — months+ (preferences, relationships, skills, life events)\n\n"
            "Score 0-10. If TRANSIENT, score MUST be 0."
        )
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "utility",
                "schema": {
                    "type": "object",
                    "properties": {
                        "lifecycle": {
                            "type": "string",
                            "enum": ["TRANSIENT", "SHORT_TERM", "PERSISTENT"]
                        },
                        "score": {"type": "integer"}
                    },
                    "required": ["lifecycle", "score"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
        try:
            raw = self.llm_controller.llm.get_completion(
                prompt, response_format=response_format, temperature=0.0
            )
            parsed = json.loads(raw)
            if parsed.get("lifecycle") == "TRANSIENT":
                return 0.0
            score = parsed.get("score", 5)
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
        self.stm_buffer: deque = deque(maxlen=200)
        self.stm_embeddings = None  # numpy array, parallel to stm_buffer
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
            # Local SentenceTransformer encode — zero LLM token cost
            emb = self.router.retriever.model.encode([content])
            if self.stm_embeddings is None:
                self.stm_embeddings = emb
            else:
                self.stm_embeddings = np.vstack([self.stm_embeddings, emb])
            # deque auto-evicts from left; keep embeddings in sync
            while self.stm_embeddings is not None and len(self.stm_embeddings) > len(self.stm_buffer):
                self.stm_embeddings = self.stm_embeddings[1:]
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
            self.bm25_corpus_tokens.append(simple_tokenize(doc.lower()))
            self._bm25_dirty = True
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
            'per_step': self.routing_stats['per_step'],
        }

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Graph retrieval (RRF fusion) + STM shadow buffer fallback."""
        # Semantic top score for confidence check
        _, scores = self.retriever.search(query, k)
        max_score = float(scores[0]) if len(scores) > 0 else 0.0

        # Parent RRF fusion + temporal sort
        memory_str = super().find_related_memories_raw(query, k)

        # STM fallback: when graph confidence is low, search skipped content
        if (max_score < 0.35 and self.stm_embeddings is not None
                and len(self.stm_buffer) > 0):
            query_emb = self.router.retriever.model.encode([query])
            stm_sims = cosine_similarity(query_emb, self.stm_embeddings)[0]
            stm_top = np.argsort(stm_sims)[-min(3, len(self.stm_buffer)):][::-1]
            stm_items = list(self.stm_buffer)
            hits = [stm_items[idx] for idx in stm_top if stm_sims[idx] > 0.2]
            if hits:
                memory_str += "\n--- Recent skipped context ---\n"
                memory_str += "\n".join(hits) + "\n"

        return memory_str

    def export_graph_json(self) -> dict:
        """Export graph with D-MEM routing metadata and STM buffer info."""
        data = super().export_graph_json()
        data['graph']['routing'] = self.get_routing_summary()
        data['graph']['stm_buffer_size'] = len(self.stm_buffer)
        return data


# Alias so we don't shadow the builtin `time` parameter name
time_module_time = time.time
