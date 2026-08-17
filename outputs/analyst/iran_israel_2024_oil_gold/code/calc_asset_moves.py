import pandas as pd
from analyst import toolkit


def run(asset_a_prices, asset_b_prices):
    anchor = pd.Timestamp('2024-04-15')
    horizons = [1, 5, 20, 60]

    rows = []
    for asset_name, df in [('WTI crude', asset_a_prices), ('Gold', asset_b_prices)]:
        series = df['Close']
        for h in horizons:
            pct = toolkit.pct_move(series, anchor, h)
            rows.append({'asset': asset_name, 'horizon_days': h, 'pct_move': pct})

    return pd.DataFrame(rows, columns=['asset', 'horizon_days', 'pct_move'])