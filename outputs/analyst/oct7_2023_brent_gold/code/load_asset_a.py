from analyst import toolkit

def run():
    df = toolkit.load_prices('BZ=F', '2023-04-12', '2024-02-06')
    df.index.name = 'market_trading_day'
    return df