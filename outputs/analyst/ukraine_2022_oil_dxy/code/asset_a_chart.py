import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

def run(asset_a_prices):
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(asset_a_prices.index, asset_a_prices['Close'], label='WTI Crude Close', color='tab:blue')

    event_date = pd.Timestamp('2022-02-24')
    ax.axvline(event_date, color='red', linestyle='--',
               label='Russian full-scale invasion of Ukraine')

    ax.set_title('WTI Crude Oil Price Around Russian Full-Scale Invasion of Ukraine')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price (USD per barrel)')
    ax.legend()

    fig.tight_layout()
    return fig