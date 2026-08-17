import pandas as pd
from analyst import toolkit

def run(asset_moves):
    df = asset_moves.copy()
    df['horizon_col'] = '+' + df['horizon_days'].astype(int).astype(str) + 'd'
    pivot = df.pivot_table(index='asset', columns='horizon_col', values='pct_move', aggfunc='first')
    pivot = pivot.round(2)

    horizons = sorted(df['horizon_days'].unique())
    horizon_cols = ['+' + str(int(h)) + 'd' for h in horizons]
    horizon_cols = [c for c in horizon_cols if c in pivot.columns]

    pivot = pivot.reset_index()
    pivot = pivot[['asset'] + horizon_cols]

    return pivot.reset_index(drop=True)