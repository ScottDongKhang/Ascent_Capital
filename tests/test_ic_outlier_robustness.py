"""IC gate must be robust to a single data-error outlier symbol.

Background (2026-06-24): a recurring ×10 price error in ONE symbol (KLAC) drove
the cross-sectional trend IC strongly negative (Pearson corrwith is dominated by
extreme values), so the IC gate zeroed the 70% trend sleeve. One corrupt symbol
should not be able to disable a sleeve. The fix winsorizes the forward-return
target cross-sectionally (per date) before computing IC.
"""
import numpy as np
import pandas as pd

from ascent.main import _winsorize_rows


def test_winsorize_rows_clips_outlier_leaves_bulk():
    """Per-row percentile winsorization pulls in an extreme outlier without
    materially moving the bulk values."""
    row = list(range(200))
    row[100] = -10_000  # data-error outlier
    df = pd.DataFrame([row], columns=[f"S{i}" for i in range(200)])
    out = _winsorize_rows(df, lower=0.01, upper=0.99)
    assert out.iloc[0, 100] > -10_000          # outlier pulled in
    assert out.iloc[0, 100] >= df.iloc[0].quantile(0.01)
    # a mid-bulk value is unchanged
    assert out.iloc[0, 50] == 50


def test_winsorized_ic_is_robust_to_one_outlier_symbol():
    """Cross-sectional IC of a bulk-aligned signal must stay positive when one
    symbol carries a huge target outlier — raw Pearson IC flips negative, the
    winsorized IC does not."""
    n_dates, n_syms = 12, 200
    syms = [f"S{i}" for i in range(n_syms)]
    # alpha ranks symbols 0..199 the same every date; clean fwd is rank-aligned
    alpha = pd.DataFrame(np.tile(np.arange(n_syms), (n_dates, 1)), columns=syms)
    fwd = alpha.copy().astype(float)
    # Highest-alpha symbol (max leverage in the cross-sectional correlation, like
    # a momentum-ranked KLAC) carries a huge data-error revert every date.
    fwd.iloc[:, -1] = -1_000_000.0

    raw_ic = alpha.corrwith(fwd, axis=1).mean()
    wins_ic = alpha.corrwith(_winsorize_rows(fwd, 0.01, 0.99), axis=1).mean()

    assert raw_ic < 0, f"expected raw IC poisoned negative, got {raw_ic}"
    assert wins_ic > 0.5, f"expected winsorized IC to recover, got {wins_ic}"
