import pandas as pd
from analyst import toolkit

def run(asset_a_prices, asset_b_prices):
    anchor = pd.Timestamp('2022-02-24')
    horizons = [1, 5, 20, 60]
    assets = {
        'WTI crude': asset_a_prices,
        'US dollar index': asset_b_prices,
    }
    rows = []
    for name, df in assets.items():
        for h in horizons:
            pct = toolkit.pct_move(df['Close'], anchor, h)
            rows.append({'asset': name, 'horizon_days': h, 'pct_move': pct})
    return pd.DataFrame(rows, columns=['asset', 'horizon_days', 'pct_move'])