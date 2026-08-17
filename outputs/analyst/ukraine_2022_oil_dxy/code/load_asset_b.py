from analyst import toolkit

def run():
    df = toolkit.load_prices('DX-Y.NYB', '2021-08-28', '2022-06-24')
    return df