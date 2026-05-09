"""
debate/disagreement_scorer.py

Measures semantic disagreement between debate agent reasoning traces
using TF-IDF cosine similarity (pure numpy — no external embedding API).

Formula: disagreement_score = 1 - mean([sim(bull,bear), sim(bull,devil), sim(bear,devil)])
  1.0 = maximum disagreement (agents are talking about completely different things)
  0.0 = pure consensus (agents are paraphrasing each other)

Interpretation labels (for logging and monitoring — NOT verdict overrides):
  < 0.30  genuine_disagreement   — agents are using distinct vocabulary
  0.30–0.70  moderate_convergence — partial vocabulary overlap
  > 0.70  soft_consensus         — agents are largely talking about the same things

The score is surfaced to the judge as an informational note only.
The judge is not instructed to change its verdict based on this score —
TF-IDF vocabulary overlap is too coarse a proxy to drive individual decisions.
"""
from __future__ import annotations

import math
import re
import logging
from collections import Counter
from typing import Dict, Tuple

log = logging.getLogger(__name__)

# Stop words to strip before building TF-IDF vectors
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "this", "that", "these", "those", "i", "we",
    "you", "it", "its", "our", "their", "as", "by", "from", "not", "no",
})


def _tokenize(text: str) -> list:
    """Lowercase, strip punctuation, remove stop words."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _tfidf_vector(tokens: list, vocab: list) -> list:
    """
    Compute a TF-IDF-weighted vector over the shared vocabulary.
    TF = term count / total tokens. IDF is not computed cross-document
    (single-document context) so this reduces to normalized TF.
    Returns a list of floats indexed by vocab position.
    """
    if not tokens:
        return [0.0] * len(vocab)
    counts = Counter(tokens)
    total  = len(tokens)
    vocab_index = {w: i for i, w in enumerate(vocab)}
    vec = [0.0] * len(vocab)
    for word, count in counts.items():
        if word in vocab_index:
            vec[vocab_index[word]] = count / total
    return vec


def _normalize(vec: list) -> list:
    """L2-normalize a vector. Returns zero vector if norm is near zero."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-9:
        return vec
    return [x / norm for x in vec]


def _cosine(a: list, b: list) -> float:
    """Dot product of two L2-normalized vectors."""
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def pairwise_similarities(
    bull_text: str,
    bear_text: str,
    devil_text: str,
) -> Dict[str, float]:
    """
    Compute cosine similarity between all three pairs of reasoning traces.

    Returns dict with keys: bull_bear, bull_devil, bear_devil.
    Each value is in [0, 1] where 1 = identical vocabulary distribution.
    """
    bull_tokens  = _tokenize(bull_text)
    bear_tokens  = _tokenize(bear_text)
    devil_tokens = _tokenize(devil_text)

    # Build shared vocabulary from union of all tokens
    vocab = sorted(set(bull_tokens) | set(bear_tokens) | set(devil_tokens))

    if not vocab:
        return {"bull_bear": 0.0, "bull_devil": 0.0, "bear_devil": 0.0}

    bull_vec  = _normalize(_tfidf_vector(bull_tokens,  vocab))
    bear_vec  = _normalize(_tfidf_vector(bear_tokens,  vocab))
    devil_vec = _normalize(_tfidf_vector(devil_tokens, vocab))

    return {
        "bull_bear":  round(_cosine(bull_vec, bear_vec),  4),
        "bull_devil": round(_cosine(bull_vec, devil_vec), 4),
        "bear_devil": round(_cosine(bear_vec, devil_vec), 4),
    }


def compute_disagreement_score(
    bull_text: str,
    bear_text: str,
    devil_text: str,
) -> float:
    """
    Compute the aggregate disagreement score across all three agent pairs.

    Returns:
        Float in [0, 1].
        1.0 = maximum disagreement (fully distinct reasoning traces)
        0.0 = pure consensus (agents are semantically identical)
    """
    if not any([bull_text.strip(), bear_text.strip(), devil_text.strip()]):
        return 0.0

    sims  = pairwise_similarities(bull_text, bear_text, devil_text)
    mean_sim = (sims["bull_bear"] + sims["bull_devil"] + sims["bear_devil"]) / 3.0
    return round(1.0 - mean_sim, 4)


def interpret_disagreement(score: float) -> str:
    """Return a human-readable label for a disagreement score."""
    if score < 0.30:
        return "genuine_disagreement"
    if score < 0.70:
        return "moderate_convergence"
    return "soft_consensus"


def format_disagreement_for_judge(disagreement_score: float) -> str:
    """
    Format the disagreement score as an informational note for the judge.

    This is observational context only — the judge is NOT instructed to
    change its verdict direction based on this score. TF-IDF vocabulary
    overlap is too coarse a proxy to override the judge's synthesis of
    the actual arguments.

    Args:
        disagreement_score: Value from compute_disagreement_score() — 1=max disagreement.

    Returns:
        A short informational line the judge can use as context.
    """
    similarity = round(1.0 - disagreement_score, 2)
    label      = interpret_disagreement(disagreement_score)

    return (
        f"[Monitoring] Agent trace similarity: {similarity:.2f} "
        f"(0.0 = fully distinct vocabulary, 1.0 = identical). "
        f"Disagreement score: {disagreement_score:.2f} ({label.replace('_', ' ')}). "
        f"This is an observational metric — use your judgment on the arguments themselves."
    )
