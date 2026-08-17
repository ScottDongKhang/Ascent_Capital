def run(asset_moves):
    import pandas as pd

    df = asset_moves.copy()
    df['pct_move'] = df['pct_move'].round(2)

    pivot = df.pivot_table(index='asset', columns='horizon_days', values='pct_move', aggfunc='first')

    # rename columns to '+Xd' format, sorted by horizon
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot.columns = [f'+{int(c)}d' for c in pivot.columns]

    pivot = pivot.reset_index()
    pivot = pivot.rename(columns={'asset': 'asset'})

    return pivot.reset_index(drop=True)