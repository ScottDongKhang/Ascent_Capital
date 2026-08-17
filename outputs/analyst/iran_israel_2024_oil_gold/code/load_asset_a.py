from analyst import toolkit

def run():
    df = toolkit.load_prices('CL=F', '2023-10-18', '2024-08-13')
    df.index.name = 'market_trading_day'
    return df