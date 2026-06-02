"""scripts/generate_mock_models.py

Generates placeholder ML model files with random weights.

In the live system, these files are produced by:
  - ascent/alpha/ml_sleeve.py      → data_cache/ml_model_{agent_id}.pkl (XGBoost)
  - ascent/strategy/conviction_gate.py → data_cache/conviction_gate_model.pkl (LogisticRegression)
  - ascent/alpha/meta_learner.py   → data_cache/sleeve_posteriors.json (Bayesian priors)

The actual trained weights are NOT published. This script writes structurally-correct
placeholder artifacts so the pipeline can import and run without crashing.

Usage:
    python scripts/generate_mock_models.py

Requires: scikit-learn, xgboost, numpy (all in requirements.txt)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ── XGBoost mock (ML sleeve) ──────────────────────────────────────────

def _make_xgboost_mock(agent_id: str) -> None:
    try:
        import xgboost as xgb
    except ImportError:
        print(f"[MockModels] xgboost not installed — skipping {agent_id}")
        return

    # 12 features, random booster with near-zero learning
    rng = np.random.default_rng(seed=0)
    X = rng.standard_normal((200, 12))
    y = rng.integers(0, 2, size=200)

    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.0,  # frozen weights — purely structural placeholder
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X, y)

    path = CACHE_DIR / f"ml_model_{agent_id}.pkl"
    with path.open("wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_names": [f"feature_{i:02d}" for i in range(12)],
                "train_date": "2026-01-01",
                "_mock": True,
            },
            f,
        )
    print(f"[MockModels] Wrote {path}")


# ── LogisticRegression mock (conviction gate) ─────────────────────────

def _make_conviction_gate_mock() -> None:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("[MockModels] scikit-learn not installed — skipping conviction gate")
        return

    rng = np.random.default_rng(seed=0)
    X = rng.standard_normal((100, 15))
    y = rng.integers(0, 2, size=100)

    model = LogisticRegression(random_state=42, max_iter=10, C=0.0)
    model.fit(X, y)
    # Zero out coefficients so the mock never makes real decisions
    model.coef_[:] = 0.0

    path = CACHE_DIR / "conviction_gate_model.pkl"
    with path.open("wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_names": [f"feat_{i}" for i in range(15)],
                "train_date": "2026-01-01",
                "n_cases": 0,
                "_mock": True,
            },
            f,
        )
    print(f"[MockModels] Wrote {path}")


# ── Bayesian sleeve posteriors mock (meta-learner) ────────────────────

def _make_sleeve_posteriors_mock() -> None:
    sleeve_names = [
        "trend", "meanrev", "volatility", "statarb", "ml", "fundamental",
        "llm_fundamental", "earnings", "analyst", "options_flow",
        "insider", "short_interest", "narrative",
    ]
    regimes = ["calm_bull", "stressed", "crisis", "uncertain"]

    # Uninformative priors: mean=0, precision=1 (equal-weight starting point)
    posteriors: dict = {}
    for regime in regimes:
        posteriors[regime] = {}
        for sleeve in sleeve_names:
            posteriors[regime][sleeve] = {
                "mu":        0.0,
                "precision": 1.0,
                "n_obs":     0,
                "_mock":     True,
            }

    path = CACHE_DIR / "sleeve_posteriors.json"
    path.write_text(json.dumps(posteriors, indent=2))
    print(f"[MockModels] Wrote {path}")


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    for agent_id in ("us_equities", "macro", "international", "alternatives"):
        _make_xgboost_mock(agent_id)
    _make_conviction_gate_mock()
    _make_sleeve_posteriors_mock()
    print("[MockModels] Done. These are structural placeholders only.")
    print("[MockModels] Real model artifacts are excluded from the public repo.")
