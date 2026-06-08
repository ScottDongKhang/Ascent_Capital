# ascent/integrations/analogue_matcher.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANALOGUES_PATH = _REPO_ROOT / "data_cache" / "mirofish_analogues.json"
_MIN_CONFIDENCE = 0.05


def _load_analogues() -> list[dict[str, Any]]:
    try:
        if _ANALOGUES_PATH.exists():
            return json.loads(_ANALOGUES_PATH.read_text())
        return []
    except Exception as exc:
        log.debug("[AnalogueMatcher] Load failed: %s", exc)
        return []


def _doc_for_analogue(a: dict) -> str:
    return " ".join([
        a.get("description", ""),
        " ".join(a.get("keywords", [])),
        " ".join(a.get("affected_sectors", [])),
        " ".join(a.get("affected_symbols", [])),
    ]).lower()


def _keyword_overlap_score(query: str, doc: str) -> float:
    q_words = set(query.lower().split())
    d_words = set(doc.lower().split())
    overlap = q_words & d_words
    if not q_words:
        return 0.0
    return len(overlap) / len(q_words)


def find_analogues(
    event_description: str,
    symbols: list[str],
    top_k: int = 3,
) -> list[tuple[dict[str, Any], float]]:
    """
    Find the top-k most similar historical analogue events.

    Returns list of (analogue_dict, confidence_0_to_1), sorted by confidence descending.
    Confidence < MIN_CONFIDENCE analogues are excluded.
    Falls back to keyword overlap if sklearn is unavailable.
    """
    analogues = _load_analogues()
    if not analogues:
        return []

    query = f"{event_description} {' '.join(symbols)}".lower()
    corpus = [_doc_for_analogue(a) for a in analogues]

    similarities: list[float] = []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        all_docs = corpus + [query]
        tfidf_matrix = vectorizer.fit_transform(all_docs)
        sims = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
        similarities = sims.tolist()
    except Exception as exc:
        log.debug("[AnalogueMatcher] TF-IDF failed (%s), using keyword overlap fallback", exc)
        similarities = [_keyword_overlap_score(query, doc) for doc in corpus]

    import numpy as np
    arr = np.array(similarities)
    top_indices = arr.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        conf = float(arr[idx])
        if conf >= _MIN_CONFIDENCE:
            results.append((analogues[idx], conf))
    return results
