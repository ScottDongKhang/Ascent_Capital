from analyst import toolkit

def run():
    df = toolkit.load_prices('CL=F', '2021-08-28', '2022-06-24')
    df.index.name = 'market_trading_day'
    return df