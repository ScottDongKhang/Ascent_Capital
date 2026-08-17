import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from analyst import toolkit

def run(asset_a_prices):
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(asset_a_prices.index, asset_a_prices['Close'],
            label='WTI Crude Close Price', color='tab:blue')

    event_date = pd.Timestamp('2024-04-15')
    ax.axvline(event_date, color='red', linestyle='--',
               label='Iranian missile and drone strike on Israel')

    ax.set_title('WTI Crude Oil Price Around Iranian Missile and Drone Strike on Israel (2024-04-15)')
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Close Price (USD per barrel)')
    ax.legend()

    fig.autofmt_xdate()

    return fig