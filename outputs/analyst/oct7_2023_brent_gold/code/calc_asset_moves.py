import pandas as pd
from analyst import toolkit

def run(asset_a_prices, asset_b_prices):
    anchor = pd.Timestamp('2023-10-09')
    horizons = [1, 5, 20, 60]
    assets = [('Brent crude', asset_a_prices), ('Gold', asset_b_prices)]

    rows = []
    for name, df in assets:
        series = df['Close']
        for h in horizons:
            pct = toolkit.pct_move(series, anchor, h)
            rows.append({'asset': name, 'horizon_days': h, 'pct_move': pct})

    return pd.DataFrame(rows, columns=['asset', 'horizon_days', 'pct_move'])