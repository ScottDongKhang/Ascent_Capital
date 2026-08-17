import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from analyst import toolkit


def run(asset_a_prices):
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(asset_a_prices.index, asset_a_prices['Close'], label='Brent Crude Close Price')

    event_date = pd.Timestamp('2023-10-09')
    ax.axvline(event_date, color='red', linestyle='--', label='Hamas attack on Israel')

    ax.set_title('Brent Crude Price Around Hamas Attack on Israel (2023-10-09)')
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Close Price (USD per barrel)')
    ax.legend()

    return fig