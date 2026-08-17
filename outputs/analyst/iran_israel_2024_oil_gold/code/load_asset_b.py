from analyst import toolkit

def run():
    df = toolkit.load_prices('GC=F', '2023-10-18', '2024-08-13')
    return df