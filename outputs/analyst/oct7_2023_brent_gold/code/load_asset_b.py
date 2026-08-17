from analyst import toolkit

def run():
    df = toolkit.load_prices('GC=F', '2023-04-12', '2024-02-06')
    return df