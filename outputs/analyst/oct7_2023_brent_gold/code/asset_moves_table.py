import pandas as pd
from analyst import toolkit

def run(asset_moves):
    df = asset_moves.copy()
    df['pct_move'] = df['pct_move'].round(2)
    pivot = df.pivot_table(index='asset', columns='horizon_days', values='pct_move', aggfunc='first')
    pivot = pivot.rename(columns=lambda h: f'+{h}d')
    horizon_order = sorted(pivot.columns, key=lambda c: int(c[1:-1]))
    pivot = pivot[horizon_order]
    pivot = pivot.reset_index()
    return pivot