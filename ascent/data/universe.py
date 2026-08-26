"""
Ascent Capital — Historical Universe
Tracks which stocks were tradeable on each date.

Option B complete (2026-05-02):
- SP500_MEMBERS: 503 current S&P 500 members from Wikipedia (as-of scrape date)
- REMOVED_STOCKS: 260 S&P 500 removals 2013–2026 from Wikipedia changes table
- SYMBOL_ADDITION_DATES: 92 post-2020 real addition dates from Wikipedia
- Symbols with no recorded addition date default to UNIVERSE_START (2020-01-01)
"""
import json
from pathlib import Path
import pandas as pd


def _load_addition_dates_from_json() -> dict:
    """Load addition dates from the universe JSON file."""
    universe_path = Path(__file__).parent.parent / "config" / "us_equity_universe.json"
    if not universe_path.exists():
        return {}
    try:
        with open(universe_path) as f:
            data = json.load(f)
        return data.get("addition_dates", {})
    except Exception:
        return {}

# S&P 500 constituent list scraped from Wikipedia (Option B, 2026-05-02).
# 503 current members. BRK.B / BF.B normalized to BRK-B / BF-B (yfinance convention).
SP500_MEMBERS = frozenset({
    "A","AAPL","ABBV","ABNB","ABT","ACGL","ACN","ADBE","ADI","ADM","ADP","ADSK",
    "AEE","AEP","AES","AFL","AIG","AIZ","AJG","AKAM","ALB","ALGN","ALL","ALLE",
    "AMAT","AMCR","AMD","AME","AMGN","AMP","AMT","AMZN","ANET","AON","AOS","APA",
    "APD","APH","APO","APP","APTV","ARE","ARES","ATO","AVB","AVGO","AVY","AWK",
    "AXON","AXP","AZO","BA","BAC","BALL","BAX","BBY","BDX","BEN","BF-B","BG",
    "BIIB","BK","BKNG","BKR","BLDR","BLK","BMY","BR","BRK-B","BRO","BSX","BX",
    "BXP","C","CAG","CAH","CARR","CASY","CAT","CB","CBOE","CBRE","CCI","CCL",
    "CDNS","CDW","CEG","CF","CFG","CHD","CHRW","CHTR","CI","CIEN","CINF","CL",
    "CLX","CMCSA","CME","CMG","CMI","CMS","CNC","CNP","COF","COHR","COIN","COO",
    "COP","COR","COST","CPAY","CPB","CPRT","CPT","CRH","CRL","CRM","CRWD","CSCO",
    "CSGP","CSX","CTAS","CTRA","CTSH","CTVA","CVNA","CVS","CVX","D","DAL","DASH",
    "DD","DDOG","DE","DECK","DELL","DG","DGX","DHI","DHR","DIS","DLR","DLTR",
    "DOC","DOV","DOW","DPZ","DRI","DTE","DUK","DVA","DVN","DXCM","EA","EBAY",
    "ECL","ED","EFX","EG","EIX","EL","ELV","EME","EMR","EOG","EPAM","EQIX",
    "EQR","EQT","ERIE","ES","ESS","ETN","ETR","EVRG","EW","EXC","EXE","EXPD",
    "EXPE","EXR","F","FANG","FAST","FCX","FDS","FDX","FE","FFIV","FICO","FIS",
    "FISV","FITB","FIX","FOX","FOXA","FRT","FSLR","FTNT","FTV","GD","GDDY","GE",
    "GEHC","GEN","GEV","GILD","GIS","GL","GLW","GM","GNRC","GOOG","GOOGL","GPC",
    "GPN","GRMN","GS","GWW","HAL","HAS","HBAN","HCA","HD","HIG","HII","HLT",
    "HON","HOOD","HPE","HPQ","HRL","HSIC","HST","HSY","HUBB","HUM","HWM","IBKR",
    "IBM","ICE","IDXX","IEX","IFF","INCY","INTC","INTU","INVH","IP","IQV","IR",
    "IRM","ISRG","IT","ITW","IVZ","J","JBHT","JBL","JCI","JKHY","JNJ","JPM",
    "KDP","KEY","KEYS","KHC","KIM","KKR","KLAC","KMB","KMI","KO","KR","KVUE",
    "L","LDOS","LEN","LH","LHX","LII","LIN","LITE","LLY","LMT","LNT","LOW",
    "LRCX","LULU","LUV","LVS","LYB","LYV","MA","MAA","MAR","MAS","MCD","MCHP",
    "MCK","MCO","MDLZ","MDT","MET","META","MGM","MKC","MLM","MMM","MNST","MO",
    "MOS","MPC","MPWR","MRK","MRNA","MRSH","MS","MSCI","MSFT","MSI","MTB","MTD",
    "MU","NCLH","NDAQ","NDSN","NEE","NEM","NFLX","NI","NKE","NOC","NOW","NRG",
    "NSC","NTAP","NTRS","NUE","NVDA","NVR","NWS","NWSA","NXPI","O","ODFL","OKE",
    "OMC","ON","ORCL","ORLY","OTIS","OXY","PANW","PAYX","PCAR","PCG","PEG","PEP",
    "PFE","PFG","PG","PGR","PH","PHM","PKG","PLD","PLTR","PM","PNC","PNR",
    "PNW","PODD","POOL","PPG","PPL","PRU","PSA","PSKY","PSX","PTC","PWR","PYPL",
    "Q","QCOM","RCL","REG","REGN","RF","RJF","RL","RMD","ROK","ROL","ROP",
    "ROST","RSG","RTX","RVTY","SATS","SBAC","SBUX","SCHW","SHW","SJM","SLB","SMCI",
    "SNA","SNDK","SNPS","SO","SOLV","SPG","SPGI","SRE","STE","STLD","STT","STX",
    "STZ","SW","SWK","SWKS","SYF","SYK","SYY","T","TAP","TDG","TDY","TECH",
    "TEL","TER","TFC","TGT","TJX","TKO","TMO","TMUS","TPL","TPR","TRGP","TRMB",
    "TROW","TRV","TSCO","TSLA","TSN","TT","TTD","TTWO","TXN","TXT","TYL","UAL",
    "UBER","UDR","UHS","ULTA","UNH","UNP","UPS","URI","USB","V","VICI","VLO",
    "VLTO","VMC","VRSK","VRSN","VRT","VRTX","VST","VTR","VTRS","VZ","WAB","WAT",
    "WBD","WDAY","WDC","WEC","WELL","WFC","WM","WMB","WMT","WRB","WSM","WST",
    "WTW","WY","WYNN","XEL","XOM","XYL","XYZ","YUM","ZBH","ZBRA","ZTS",
})

# 260 S&P 500 removals 2013–2026, sourced from Wikipedia changes table (Option B).
# Each entry: (symbol, sector, removal_date, reason)
REMOVED_STOCKS = [
    ("FII", "Unknown", "2013-01-02", "ABBV spun off from Abbott Labs (ABT)"),
    ("BIG", "Unknown", "2013-02-15", "Market capitalization changes"),
    ("PCS", "Unknown", "2013-04-30", "A majority of MetroPCS was acquired by T-Mobile"),
    ("CVH", "Unknown", "2013-05-08", "Acquired by Aetna (AET)"),
    ("DF", "Unknown", "2013-05-23", "DF too small after spinoff of White Wave Foods"),
    ("HNZ", "Unknown", "2013-06-06", "HNZ acquired by consortium"),
    ("FHN", "Financial Services", "2013-06-21", "Zoetis spun off by Pfizer"),
    ("APOL", "Unknown", "2013-07-01", "Apollo Group's market cap more representative of a mid-cap"),
    ("S", "Unknown", "2013-07-08", "Softbank consortium purchase results in public float below 50%"),
    ("BMC", "Unknown", "2013-09-10", "BMC taken private by consortium"),
    ("SAI", "Unknown", "2013-09-20", "Market capitalization change."),
    ("NYX", "Unknown", "2013-11-13", "ICE Exchange acquired NYSE Euronext."),
    ("JCP", "Unknown", "2013-12-02", "Allegion spun off by Ingersoll Rand."),
    ("MOLX", "Unknown", "2013-12-10", "MOLX acquired by Koch Industries."),
    ("ANF", "Consumer Cyclical", "2013-12-23", "Market capitalization changes."),
    ("JDSU", "Unknown", "2013-12-23", "Market capitalization changes."),
    ("LIFE", "Unknown", "2014-01-24", "Life Technologies acquired by Thermo Fisher Scientific Inc"),
    ("WPX", "Unknown", "2014-03-21", "Market capitalization changes"),
    ("CLF", "Basic Materials", "2014-04-02", "Market capitalization changes"),
    ("BEAM", "Unknown", "2014-05-01", "Beam acquired by Suntory"),
    ("SLM", "Financial Services", "2014-05-01", "Navient Spun off from SLM"),
    ("LSI", "Unknown", "2014-05-08", "Avago acquires LSI"),
    ("IGT", "Unknown", "2014-06-20", "Market capitalization changes"),
    ("FRX", "Unknown", "2014-07-01", "Actavis acquires Forest Laboratories"),
    ("X", "Unknown", "2014-07-02", "Market capitalization changes"),
    ("RDC", "Unknown", "2014-08-18", "Market cap changes"),
    ("GHC", "Consumer Defensive", "2014-09-20", "Market cap changes"),
    ("BTU", "Unknown", "2014-09-20", "Market cap changes"),
    ("SWY", "Unknown", "2015-01-27", "Safeway acquired by private equity consortium"),
    ("COV", "Unknown", "2015-01-27", "Covidien acquired by Medtronic"),
    ("PETM", "Unknown", "2015-03-12", "PetSmart acquired by private equity consortium"),
    ("CFN", "Unknown", "2015-03-18", "Carefusion acquired by Becton Dickinson"),
    ("DNR", "Unknown", "2015-03-23", "Market Capitalization Changes."),
    ("AVP", "Unknown", "2015-03-23", "Market Capitalization Changes."),
    ("NBR", "Unknown", "2015-03-23", "Market Capitalization Changes."),
    ("WIN", "Unknown", "2015-04-07", "Market capitalization change."),
    ("LO", "Unknown", "2015-06-11", "Lorillard gets acquired."),
    ("QEP", "Unknown", "2015-07-01", "Baxalta spun off by Baxter."),
    ("TEG", "Unknown", "2015-07-01", "Integrys taken over."),
    ("ATI", "Industrials", "2015-07-02", "Spin off of Columbia Pipeline."),
    ("KRFT", "Unknown", "2015-07-06", "Kraft merger with Heinz."),
    ("FDO", "Unknown", "2015-07-08", "Family Dollar acquired."),
    ("NE", "Unknown", "2015-07-20", "PayPal Spun off by eBay"),
    ("DTV", "Unknown", "2015-07-29", "DirecTV acquired by AT&T."),
    ("PLL", "Unknown", "2015-08-28", "Pall taken over"),
    ("HSP", "Unknown", "2015-09-02", "Hospira taken over"),
    ("JOY", "Unknown", "2015-10-07", "Market capitalization change."),
    ("HCBK", "Unknown", "2015-11-02", "HPQ spins off HPE"),
    ("GNW", "Unknown", "2015-11-18", "GE spinning off Synchrony Financial"),
    ("SIAL", "Unknown", "2015-11-19", "Sigma-Aldrich acquired by Merck KGaA (MKGAY)"),
    ("CSC", "Unknown", "2015-12-01", "CSC spins off CSRA"),
    ("CMCSK", "Unknown", "2015-12-15", "CMCSK shares no longer listed"),
    ("ALTR", "Unknown", "2015-12-29", "ALTR taken over by INTC"),
    ("FOSL", "Unknown", "2016-01-05", "WSH merges with TW"),
    ("ACE", "Unknown", "2016-01-19", "ACE Ltd acquires Chubb, retains CB ticker"),
    ("BRCM", "Unknown", "2016-02-01", "FRT replaces BRCM"),
    ("PCP", "Unknown", "2016-02-01", "CFG replaces PCP"),
    ("PCL", "Unknown", "2016-02-22", "PCL taken over by WY"),
    ("CNX", "Energy", "2016-03-04", "AWK replaces CNX"),
    ("GMCR", "Unknown", "2016-03-07", "JAB Holding Company acquired Keurig Green Mountain"),
    ("ESV", "Unknown", "2016-03-30", "Centene acquired Health Net."),
    ("POM", "Unknown", "2016-03-30", "Exelon acquired Pepco."),
    ("CAM", "Unknown", "2016-04-04", "Schlumberger acquired Cameron"),
    ("THC", "Healthcare", "2016-04-18", "Ulta replaces Tenet"),
    ("GME", "Consumer Cyclical", "2016-04-25", "Global Payments acquiring Heartland Payment Systems"),
    ("ADT", "Unknown", "2016-05-03", "ADT acquired by APO"),
    ("TWC", "Unknown", "2016-05-18", "TWC acquired by CHTR"),
    ("ARG", "Unknown", "2016-05-23", "ARG acquired by Air Liquide"),
    ("CCE", "Unknown", "2016-05-31", "CCE merging with European bottlers"),
    ("BXLT", "Unknown", "2016-06-03", "SHPG acquiring BXLT"),
    ("CVC", "Unknown", "2016-06-22", "CVC acquired by Altice NV"),
    ("TE", "Unknown", "2016-07-01", "TE acquired by EMA"),
    ("GAS", "Unknown", "2016-07-01", "GAS acquired by SO"),
    ("CPGX", "Unknown", "2016-07-05", "CPGX acquired by TRP"),
    ("TYC", "Unknown", "2016-09-06", "TYC acquires JCI"),
    ("EMC", "Unknown", "2016-09-08", "Dell acquires EMC"),
    ("HOT", "Unknown", "2016-09-22", "MAR acquires HOT"),
    ("DO", "Unknown", "2016-09-30", "COTY replaces DO"),
    ("AA", "Basic Materials", "2016-11-01", "AA spins off ARNC"),
    ("OI", "Unknown", "2016-12-02", "Market capitalization change."),
    ("LM", "Unknown", "2016-12-02", "Market capitalization change."),
    ("STJ", "Unknown", "2017-01-05", "Abbott Laboratories acquired St. Jude Medical."),
    ("SE", "Unknown", "2017-02-28", "SE acquired by ENB"),
    ("PBI", "Unknown", "2017-03-01", "CBOE acquires BATS"),
    ("ENDP", "Unknown", "2017-03-02", "REG acquires EQY"),
    ("LLTC", "Unknown", "2017-03-13", "Analog Devices acquired Linear Technology."),
    ("HAR", "Unknown", "2017-03-16", "Samsung Electronics acquired Harman International Industries."),
    ("FTR", "Unknown", "2017-03-20", "Market capitalization change."),
    ("URBN", "Unknown", "2017-03-20", "Market capitalization change."),
    ("SWN", "Unknown", "2017-04-04", "HPE spins off Everett, merged with CSC to form DXC"),
    ("DNB", "Unknown", "2017-04-05", "IT acquiring CEB"),
    ("TGNA", "Unknown", "2017-06-02", "TGNA spins off Cars.com"),
    ("TDC", "Unknown", "2017-06-19", "Market capitalization changes."),
    ("MJN", "Unknown", "2017-06-19", "Reckitt Benckiser acquired Mead Johnson Nutrition."),
    ("R", "Industrials", "2017-06-19", "Market capitalization changes."),
    ("YHOO", "Unknown", "2017-06-19", "VZ acquired YHOO operations"),
    ("BHI", "Unknown", "2017-07-07", "Baker Hughes merged with GE Oil & Gas."),
    ("RIG", "Unknown", "2017-07-26", "Market capitalization change."),
    ("BBBY", "Unknown", "2017-07-26", "Market capitalization change."),
    ("MNK", "Unknown", "2017-07-26", "Market capitalization change."),
    ("RAI", "Unknown", "2017-07-26", "British American Tobacco acquired Reynolds American."),
    ("MUR", "Energy", "2017-07-26", "Market capitalization change."),
    ("AN", "Consumer Cyclical", "2017-08-08", "BHF replaced AN."),
    ("WFM", "Unknown", "2017-08-29", "Amazon acquired Whole Foods Market."),
    ("SPLS", "Unknown", "2017-09-18", "Sycamore Partners acquired Staples."),
    ("LVLT", "Unknown", "2017-10-13", "CenturyLink acquired LVLT."),
    ("BCR", "Unknown", "2018-01-03", "Becton Dickinson acquired BCR."),
    ("SNI", "Unknown", "2018-03-07", "Discovery Communications acquired SNI."),
    ("CHK", "Unknown", "2018-03-19", "Market capitalization change."),
    ("PDCO", "Unknown", "2018-03-19", "Market capitalization change."),
    ("SIG", "Unknown", "2018-03-19", "Market capitalization change."),
    ("CSRA", "Unknown", "2018-04-04", "General Dynamics acquired CSRA."),
    ("WYN", "Unknown", "2018-05-31", "Wyndham Worldwide spun off Wyndham Hotels & Resorts."),
    ("NAVI", "Unknown", "2018-06-05", "Westar Energy acquired Great Plains Energy."),
    ("MON", "Unknown", "2018-06-07", "Bayer acquired Monsanto."),
    ("AYI", "Industrials", "2018-06-18", "Market capitalization change."),
    ("RRC", "Energy", "2018-06-18", "Market capitalization change."),
    ("TWX", "Unknown", "2018-06-20", "AT&T acquired Time Warner."),
    ("DPS", "Unknown", "2018-07-02", "DPS acquired by Keurig Green Mountain."),
    ("GGP", "Unknown", "2018-08-28", "GGP acquired by Brookfield Property Partners."),
    ("XL", "Unknown", "2018-09-14", "XL acquired by Axa."),
    ("ANDV", "Unknown", "2018-10-01", "ANDV acquired by Marathon Petroleum."),
    ("EVHC", "Unknown", "2018-10-11", "EVHC acquired by KKR."),
    ("CA", "Unknown", "2018-11-06", "CA acquired by Broadcom."),
    ("SRCL", "Unknown", "2018-12-03", "Market Capitalization change"),
    ("AET", "Unknown", "2018-12-03", "CVS acquired Aetna"),
    ("COL", "Unknown", "2018-12-03", "UTX acquired COL"),
    ("ESRX", "Unknown", "2018-12-24", "Cigna acquired ESRX"),
    ("SCG", "Unknown", "2019-01-02", "Dominion Energy acquired SCANA Corporation"),
    ("NFX", "Unknown", "2019-02-15", "ECA acquired NFX"),
    ("GT", "Consumer Cyclical", "2019-02-27", "WAB acquired GE transportation business"),
    ("BHF", "Financial Services", "2019-04-02", "DOW spun off from DWDP"),
    ("DWDP", "Unknown", "2019-06-03", "DWDP changed to DD after spinning off Corteva."),
    ("FLR", "Industrials", "2019-06-03", "CTVA spun off from DWDP"),
    ("MAT", "Consumer Cyclical", "2019-06-07", "Amcor acquired Bemis."),
    ("BMS", "Unknown", "2019-06-11", "BMS changed to AMCR post merger with Amcor"),
    ("LLL", "Unknown", "2019-07-01", "L3 purchased by Harris Corporation"),
    ("RHT", "Unknown", "2019-07-15", "IBM acquired Red Hat."),
    ("FL", "Unknown", "2019-08-09", "Market capitalization change."),
    ("APC", "Unknown", "2019-08-09", "Occidental Petroleum acquired Anadarko"),
    ("TSS", "Unknown", "2019-09-23", "Global Payments acquired TSS."),
    ("JEF", "Financial Services", "2019-09-26", "JEF spun off SPB."),
    ("NKTR", "Unknown", "2019-10-03", "Market capitalization change."),
    ("CELG", "Unknown", "2019-11-21", "Bristol-Myers Squibb acquired Celgene."),
    ("VIAB", "Unknown", "2019-12-05", "CBS acquired Viacom to form ViacomCBS."),
    ("STI", "Unknown", "2019-12-09", "BB&T acquired SunTrust to form Truist Financial."),
    ("AMG", "Financial Services", "2019-12-23", "Market capitalization change."),
    ("MAC", "Unknown", "2019-12-23", "Market capitalization change."),
    ("TRIP", "Unknown", "2019-12-23", "Market capitalization change."),
    ("WCG", "Unknown", "2020-01-28", "Centene Corp. acquired Wellcare Health Plans."),
    # KNOWN DATA-QUALITY ISSUE (not fixed here, out of scope for this pass): XEC's reason
    # string below describes the Gardner Denver / Ingersoll Rand industrial-businesses deal,
    # which has nothing to do with XEC (Cimarex Energy) — a ticker/reason-string mismatch
    # inherited from the Wikipedia scrape. Left untouched intentionally; flag for a future pass.
    ("XEC", "Unknown", "2020-03-02", "Gardner Denver acquired Ingersoll Rand's industrial businesses."),
    ("ARNC", "Unknown", "2020-04-01", "Arconic separated into 2 companies."),
    ("RTN", "Unknown", "2020-04-06", "United Technologies acquired Raytheon Company."),
    ("M", "Consumer Cyclical", "2020-04-06", "Market capitalization change."),
    ("AGN", "Unknown", "2020-05-12", "Allergan acquired by AbbVie."),
    ("CPRI", "Consumer Cyclical", "2020-05-12", "Market capitalization change."),
    ("HP", "Energy", "2020-05-22", "Market capitalization change."),
    ("JWN", "Unknown", "2020-06-22", "Market capitalization change."),
    ("HOG", "Consumer Cyclical", "2020-06-22", "Market capitalization change."),
    ("ADS", "Unknown", "2020-06-22", "Market capitalization change."),
    ("COTY", "Consumer Defensive", "2020-09-21", "Market capitalization change."),
    ("KSS", "Unknown", "2020-09-21", "Market capitalization change."),
    ("HRB", "Consumer Cyclical", "2020-09-21", "Market capitalization change."),
    ("ETFC", "Unknown", "2020-10-07", "Morgan Stanley acquired E*Trade."),
    ("NBL", "Unknown", "2020-10-12", "Chevron acquired Noble Energy."),
    ("AIV", "Unknown", "2020-12-21", "Apartment Investment and Management spun off Apartment Income REIT."),
    ("TIF", "Consumer Cyclical", "2021-01-07", "LVMH acquired Tiffany & Co."),
    ("CXO", "Unknown", "2021-01-21", "ConocoPhillips acquired Concho Resources."),
    ("FTI", "Energy", "2021-02-12", "TechnipFMC removed in anticipation of spin-off."),
    ("VNT", "Unknown", "2021-03-22", "Market capitalization change."),
    ("XRX", "Industrials", "2021-03-22", "Market capitalization change."),
    ("SLG", "Unknown", "2021-03-22", "Market capitalization change."),
    ("FLS", "Industrials", "2021-03-22", "Market capitalization change."),
    ("VAR", "Unknown", "2021-04-20", "Siemens Healthineers acquired Varian Medical Systems."),
    ("FLIR", "Unknown", "2021-05-14", "Teledyne Technologies acquired FLIR Systems."),
    ("HFC", "Unknown", "2021-06-04", "Market capitalization change."),
    ("ALXN", "Unknown", "2021-07-21", "AstraZeneca acquired Alexion Pharmaceuticals."),
    ("MXIM", "Unknown", "2021-08-30", "Analog Devices acquired Maxim Integrated Products."),
    ("PRGO", "Unknown", "2021-09-20", "Market capitalization change."),
    ("NOV", "Energy", "2021-09-20", "Market capitalization change."),
    ("UNM", "Financial Services", "2021-09-20", "Market capitalization change."),
    ("KSU", "Unknown", "2021-12-14", "Canadian Pacific acquired Kansas City Southern."),
    ("WU", "Unknown", "2021-12-20", "Market capitalization change."),
    ("HBI", "Unknown", "2021-12-20", "Market capitalization change."),
    ("LEG", "Unknown", "2021-12-20", "Market capitalization change."),
    ("GPS", "Unknown", "2022-02-03", "Market capitalization change."),
    ("XLNX", "Unknown", "2022-02-15", "Advanced Micro Devices acquired Xilinx."),
    ("INFO", "Unknown", "2022-03-02", "S&P Global Inc. acquired IHS Markit."),
    ("PBCT", "Unknown", "2022-04-04", "M&T Bank Corp. acquired People's United Financial."),
    ("DISCK", "Unknown", "2022-04-11", "WarnerMedia and Discovery merge to create Warner Bros. Discovery."),
    ("DISCA", "Unknown", "2022-04-11", "WarnerMedia and Discovery merge to create Warner Bros. Discovery."),
    ("CERN", "Unknown", "2022-06-08", "Oracle Corp. acquired Cerner."),
    ("IPGP", "Technology", "2022-06-21", "Market capitalization change."),
    ("UAA", "Unknown", "2022-06-21", "Market capitalization change."),
    ("UA", "Consumer Cyclical", "2022-06-21", "Market capitalization change."),
    ("PENN", "Unknown", "2022-09-19", "Market capitalization change."),
    ("PVH", "Consumer Cyclical", "2022-09-19", "Market capitalization change."),
    ("DRE", "Unknown", "2022-10-03", "Prologis Inc. acquired Duke Realty."),
    ("CTXS", "Unknown", "2022-10-03", "Vista Equity Partners acquired Citrix Systems."),
    ("NLSN", "Unknown", "2022-10-12", "Elliot Management Corp acquired Nielsen Holdings."),
    ("TWTR", "Unknown", "2022-11-01", "Elon Musk acquired Twitter."),
    ("MBC", "Unknown", "2022-12-19", "Market capitalization change."),
    ("FBHS", "Unknown", "2022-12-19", "Market capitalization change."),
    ("ABMD", "Unknown", "2022-12-22", "Johnson & Johnson acquired Abiomed."),
    ("VNO", "Real Estate", "2023-01-05", "Market capitalization change."),
    ("SIVB", "Unknown", "2023-03-15", "FDIC placed Silicon Valley Bank into receivership."),
    ("SBNY", "Unknown", "2023-03-15", "FDIC placed Signature Bank into receivership."),
    ("LUMN", "Communication Services", "2023-03-20", "Market capitalization change."),
    ("FRC", "Unknown", "2023-05-04", "FDIC placed First Republic Bank into receivership."),
    ("DISH", "Unknown", "2023-06-20", "Market capitalization change."),
    ("AAP", "Unknown", "2023-08-25", "Market capitalization change."),
    ("LNC", "Unknown", "2023-09-18", "Market capitalization change."),
    ("NWL", "Unknown", "2023-09-18", "Market capitalization change."),
    ("DXC", "Unknown", "2023-10-03", "Market capitalization change."),
    ("ATVI", "Unknown", "2023-10-18", "Microsoft acquired Activision Blizzard."),
    ("OGN", "Unknown", "2023-10-18", "Market capitalization change."),
    ("SEDG", "Technology", "2023-12-18", "Market capitalization change."),
    ("ALK", "Industrials", "2023-12-18", "Market capitalization change."),
    ("SEE", "Unknown", "2023-12-18", "Market capitalization change."),
    ("ZION", "Unknown", "2024-03-18", "Market capitalization change."),
    ("WHR", "Unknown", "2024-03-18", "Market capitalization change."),
    ("VFC", "Unknown", "2024-04-03", "Market capitalization change."),
    ("XRAY", "Healthcare", "2024-04-03", "Market capitalization change."),
    ("PXD", "Unknown", "2024-05-08", "ExxonMobil acquired Pioneer Natural Resources."),
    ("RHI", "Unknown", "2024-06-24", "Market capitalization change."),
    ("ILMN", "Healthcare", "2024-06-24", "Market capitalization change."),
    ("CMA", "Unknown", "2024-06-24", "Market capitalization change."),
    ("AAL", "Industrials", "2024-09-23", "Market capitalization change."),
    ("ETSY", "Unknown", "2024-09-23", "Market capitalization change."),
    ("BIO", "Healthcare", "2024-09-23", "Market capitalization change."),
    ("BBWI", "Consumer Cyclical", "2024-10-01", "Market capitalization change."),
    ("MRO", "Unknown", "2024-11-26", "ConocoPhillips acquired Marathon Oil."),
    ("CTLT", "Unknown", "2024-12-23", "Novo Holdings A/S acquired Catalent."),
    ("AMTM", "Unknown", "2024-12-23", "Market capitalization change."),
    ("QRVO", "Unknown", "2024-12-23", "Market capitalization change."),
    ("FMC", "Unknown", "2025-03-24", "Market capitalization change."),
    ("CE", "Unknown", "2025-03-24", "Market capitalization change."),
    ("TFX", "Unknown", "2025-03-24", "Market capitalization change."),
    ("BWA", "Consumer Cyclical", "2025-03-24", "Market capitalization change."),
    ("DFS", "Unknown", "2025-05-19", "Capital One Financial Corp. acquired Discover Financial Services."),
    ("JNPR", "Unknown", "2025-07-09", "Hewlett Packard Enterprise Co. acquired Juniper Networks."),
    ("ANSS", "Unknown", "2025-07-18", "Synopsys Inc. acquired Ansys."),
    ("HES", "Unknown", "2025-07-23", "Chevron Corp. acquired Hess Corporation"),
    ("WBA", "Unknown", "2025-08-28", "Sycamore Partners acquired Walgreen Boots Alliance."),
    ("ENPH", "Unknown", "2025-09-22", "Market capitalization change."),
    ("CZR", "Unknown", "2025-09-22", "Market capitalization change."),
    ("MKTX", "Unknown", "2025-09-22", "Market capitalization change."),
    ("KMX", "Unknown", "2025-10-31", "Market capitalization change."),
    ("EMN", "Unknown", "2025-11-04", "Market capitalization change."),
    ("IPG", "Unknown", "2025-11-28", "Omnicom Group Inc. acquired Interpublic Group."),
    ("K", "Unknown", "2025-12-11", "Mars Inc. acquired Kellanova."),
    ("MHK", "Unknown", "2025-12-22", "Market capitalization change."),
    ("SOLS", "Basic Materials", "2025-12-22", "Market capitalization change."),
    ("LKQ", "Unknown", "2025-12-22", "Market capitalization change."),
    ("DAY", "Unknown", "2026-02-09", "Thoma Bravo L.P. acquired Dayforce."),
    ("MOH", "Unknown", "2026-03-23", "Market capitalization change."),
    ("MTCH", "Unknown", "2026-03-23", "Market capitalization change."),
    ("PAYC", "Unknown", "2026-03-23", "Market capitalization change."),
    ("LW", "Unknown", "2026-03-23", "Market capitalization change."),
    ("HOLX", "Unknown", "2026-04-09", "Blackstone Inc. and TPG Inc. acquired Hologic."),
]

# ---------------------------------------------------------------------------
# DELISTING_TERMINAL_TERMS — real terminal-value deal terms for a bounded,
# already-sourced subset of REMOVED_STOCKS entries.
#
# Why this exists: without it, a delisted name simply vanishes from the
# walk-forward return calc the day its price data stops (ascent/main.py's
# "likely delisted" path / ascent/research/walk_forward_runner.py's
# close_full NaN -> fillna(0) behavior) instead of being credited the real
# cash/stock consideration shareholders actually received. That silently
# inflates OOS performance whenever a held position happens to be a name
# that got acquired. This dict lets walk_forward_runner.py credit the real
# terminal return, for exactly these 12 names, instead of the silent drop.
#
# Deliberately a SEPARATE dict, not a change to REMOVED_STOCKS: multiple call
# sites unpack REMOVED_STOCKS positionally as
#   `for sym, sector, removed, reason in REMOVED_STOCKS:`
# so changing that tuple's shape would break them. Keying a parallel dict by
# symbol avoids touching that contract at all.
#
# Date handling: `close_date` on each entry is the deal-close date verified
# against SEC filings / acquirer press releases (per-entry `source` note).
# `removed_date` mirrors the REMOVED_STOCKS date for the same symbol above —
# used as the actual "position closure" date so this stays consistent with
# build_historical_universe()'s existing end_date machinery (removed-status
# rows there use the REMOVED_STOCKS date verbatim as end_date; see
# build_historical_universe()). The two dates usually differ by a few
# trading days (REMOVED_STOCKS records when Wikipedia's S&P 500 changes
# table reflects the index removal, a few days after or around the actual
# deal close) — that gap is intentional and documented per-entry below, not
# silently collapsed to one date. K's close_date additionally carries a
# reported source discrepancy (12-11 vs 12-15/12-18 across outlets); the
# file's own REMOVED_STOCKS date (12-11) is used as the operational date.
#
# Deal types:
#   "cash"          — cash_amount is the flat per-share consideration.
#   "stock"         — acquirer_symbol + exchange_ratio; per-share value is
#                      ratio * acquirer's close price on/near close_date,
#                      looked up from actual price data at runtime (never
#                      hardcoded here) so the credited value stays internally
#                      consistent with whatever price series is in use.
#   "cash_and_stock" — both of the above, summed.
#
# Any non-tradable contingent-value-right upside (WBA's VillageMD right,
# HOLX's CVR) is explicitly NOT modeled — excluded from cash_amount below —
# because its payout is uncertain/unbounded and crediting it would itself be
# a form of look-ahead-flavored optimism. See CLAUDE.md integrity constraint
# #1 and the look-ahead reasoning note beside the walk_forward_runner.py
# credit logic for why this is a same-scope extension of that constraint,
# not a departure from it.
DELISTING_TERMINAL_TERMS = {
    "PXD": {
        "deal_type":       "stock",
        "acquirer_symbol": "XOM",
        "exchange_ratio":  2.3234,
        "cash_amount":     0.0,
        "close_date":      "2024-05-03",
        "removed_date":    "2024-05-08",
        "source":          "ExxonMobil press release; CNBC; SEC 424B3",
    },
    "MRO": {
        "deal_type":       "stock",
        "acquirer_symbol": "COP",
        "exchange_ratio":  0.2550,
        "cash_amount":     0.0,
        "close_date":      "2024-11-22",
        "removed_date":    "2024-11-26",
        "source":          "ConocoPhillips press release; SEC 8-K",
    },
    "CTLT": {
        "deal_type":       "cash",
        "acquirer_symbol": None,
        "exchange_ratio":  0.0,
        "cash_amount":     63.50,
        "close_date":      "2024-12-18",
        "removed_date":    "2024-12-23",
        "source":          "Novo Holdings press release; SEC 8-K",
    },
    "DFS": {
        "deal_type":       "stock",
        "acquirer_symbol": "COF",
        "exchange_ratio":  1.0192,
        "cash_amount":     0.0,
        "close_date":      "2025-05-18",
        "removed_date":    "2025-05-19",
        "source":          "Capital One press release; SEC S-4/A",
    },
    "JNPR": {
        "deal_type":       "cash",
        "acquirer_symbol": None,
        "exchange_ratio":  0.0,
        "cash_amount":     40.00,
        "close_date":      "2025-07-02",
        "removed_date":    "2025-07-09",
        "source":          "HPE press release",
    },
    "ANSS": {
        "deal_type":       "cash_and_stock",
        "acquirer_symbol": "SNPS",
        "exchange_ratio":  0.3399,
        "cash_amount":     199.91,
        "close_date":      "2025-07-17",
        "removed_date":    "2025-07-18",
        "source":          "Synopsys investor relations",
    },
    "HES": {
        "deal_type":       "stock",
        "acquirer_symbol": "CVX",
        "exchange_ratio":  1.025,
        "cash_amount":     0.0,
        "close_date":      "2025-07-18",
        "removed_date":    "2025-07-23",
        "source":          "Chevron press release",
    },
    "WBA": {
        "deal_type":       "cash",
        "acquirer_symbol": None,
        "exchange_ratio":  0.0,
        "cash_amount":     11.45,
        "close_date":      "2025-08-28",
        "removed_date":    "2025-08-28",
        "source":          "Sycamore Partners; Walgreens 8-K. Excludes non-tradable "
                            "contingent VillageMD right (up to $3 more) — not modeled, "
                            "too uncertain.",
    },
    "IPG": {
        "deal_type":       "stock",
        "acquirer_symbol": "OMC",
        "exchange_ratio":  0.344,
        "cash_amount":     0.0,
        "close_date":      "2025-11-26",
        "removed_date":    "2025-11-28",
        "source":          "Omnicom press release",
    },
    "K": {
        "deal_type":       "cash",
        "acquirer_symbol": None,
        "exchange_ratio":  0.0,
        "cash_amount":     83.50,
        "close_date":      "2025-12-11",
        "removed_date":    "2025-12-11",
        "source":          "Mars press release; multiple financial news sources. Some "
                            "sources report close as 12-15/12-18 instead — minor "
                            "discrepancy, the file's own REMOVED_STOCKS date (12-11) is "
                            "used as the operational date.",
    },
    "DAY": {
        "deal_type":       "cash",
        "acquirer_symbol": None,
        "exchange_ratio":  0.0,
        "cash_amount":     70.00,
        "close_date":      "2026-02-04",
        "removed_date":    "2026-02-09",
        "source":          "Thoma Bravo press release; SEC 8-K",
    },
    "HOLX": {
        "deal_type":       "cash",
        "acquirer_symbol": None,
        "exchange_ratio":  0.0,
        "cash_amount":     76.00,
        "close_date":      "2026-04-07",
        "removed_date":    "2026-04-09",
        "source":          "Blackstone press release; Hologic press release. Excludes "
                            "non-tradable CVR (up to $3 more) — not modeled, too uncertain.",
    },
}

# Real S&P 500 addition dates for symbols added after 2020-01-01.
# Source: Wikipedia List_of_S&P_500_companies (scraped Option B, 2026-05-02).
# Symbols NOT listed here default to 2020-01-01 (assumed pre-2020 member).
SYMBOL_ADDITION_DATES = {
    "IR": "2020-03-03",
    "CARR": "2020-04-03",
    "OTIS": "2020-04-03",
    "DXCM": "2020-05-12",
    "DPZ": "2020-05-12",
    "WST": "2020-05-22",
    "TDY": "2020-06-22",
    "TYL": "2020-06-22",
    "TER": "2020-09-21",
    "POOL": "2020-10-07",
    "TSLA": "2020-12-21",
    "TRMB": "2021-01-21",
    "MPWR": "2021-02-12",
    "GNRC": "2021-03-22",
    "NXPI": "2021-03-22",
    "PTC": "2021-04-20",
    "CRL": "2021-05-14",
    "MRNA": "2021-07-21",
    "TECH": "2021-08-30",
    "BRO": "2021-09-20",
    "EPAM": "2021-12-14",
    "FDS": "2021-12-20",
    "CEG": "2022-02-02",
    "NDSN": "2022-02-15",
    "CPT": "2022-04-04",
    "WBD": "2022-04-11",
    "VICI": "2022-06-08",
    "KDP": "2022-06-21",
    "ON": "2022-06-21",
    "CSGP": "2022-09-19",
    "INVH": "2022-09-19",
    "EQT": "2022-10-03",
    "PCG": "2022-10-03",
    "TRGP": "2022-10-12",
    "ACGL": "2022-11-01",
    "FSLR": "2022-12-19",
    "STLD": "2022-12-22",
    "GEHC": "2023-01-04",
    "BG": "2023-03-15",
    "PODD": "2023-03-15",
    "FICO": "2023-03-20",
    "AXON": "2023-05-04",
    "PANW": "2023-06-20",
    "KVUE": "2023-08-25",
    "ABNB": "2023-09-18",
    "BX": "2023-09-18",
    "VLTO": "2023-10-02",
    "HUBB": "2023-10-18",
    "LULU": "2023-10-18",
    "BLDR": "2023-12-18",
    "JBL": "2023-12-18",
    "UBER": "2023-12-18",
    "DECK": "2024-03-18",
    "SMCI": "2024-03-18",
    "SOLV": "2024-04-01",
    "GEV": "2024-04-02",
    "VST": "2024-05-08",
    "CRWD": "2024-06-24",
    "GDDY": "2024-06-24",
    "KKR": "2024-06-24",
    "SW": "2024-07-08",
    "DELL": "2024-09-23",
    "ERIE": "2024-09-23",
    "PLTR": "2024-09-23",
    "TPL": "2024-11-26",
    "APO": "2024-12-23",
    "LII": "2024-12-23",
    "WDAY": "2024-12-23",
    "DASH": "2025-03-24",
    "EXE": "2025-03-24",
    "TKO": "2025-03-24",
    "WSM": "2025-03-24",
    "COIN": "2025-05-19",
    "DDOG": "2025-07-09",
    "TTD": "2025-07-18",
    "XYZ": "2025-07-23",
    "IBKR": "2025-08-28",
    "APP": "2025-09-22",
    "EME": "2025-09-22",
    "HOOD": "2025-09-22",
    "Q": "2025-11-03",
    "SNDK": "2025-11-28",
    "ARES": "2025-12-11",
    "CVNA": "2025-12-22",
    "FIX": "2025-12-22",
    "CRH": "2025-12-22",
    "CIEN": "2026-02-09",
    "COHR": "2026-03-23",
    "SATS": "2026-03-23",
    "LITE": "2026-03-23",
    "VRT": "2026-03-23",
    "CASY": "2026-04-09",
}

# Default start of backtest universe
UNIVERSE_START = "2020-01-01"
UNIVERSE_END   = "2099-12-31"


def build_historical_universe(strict: bool = False, sp500_only: bool = False):
    """
    Build a DataFrame with symbol, sector, start_date, end_date.

    strict=True: exclude symbols without a recorded addition date rather than
    defaulting to UNIVERSE_START. Use in walk-forward runs.

    sp500_only=True: restrict to SP500_MEMBERS + REMOVED_STOCKS only.
    After Option B this is the recommended mode — all 503 S&P 500 members have
    real addition dates and 260 removals are tracked back to 2013.

    Addition dates merged from: SYMBOL_ADDITION_DATES (hardcoded) then
    ascent/config/us_equity_universe.json (JSON wins on conflict).
    """
    records = []

    try:
        from ascent.config.settings import get_config
        cfg             = get_config()
        current_symbols = cfg.universe.symbols
    except Exception:
        current_symbols = []

    sector_map = {}
    try:
        from ascent.data.store.parquet import load_parquet, has_data
        if has_data("profiles"):
            profiles   = load_parquet("profiles")
            sector_map = dict(zip(profiles["symbol"], profiles["sector"]))
    except Exception:
        pass

    # Merge addition dates: hardcoded dict + JSON file (JSON wins)
    merged_addition_dates = {**SYMBOL_ADDITION_DATES, **_load_addition_dates_from_json()}

    removed_set = {sym for sym, *_ in REMOVED_STOCKS}

    _missing_dates = []

    for sym in current_symbols:
        if sym in removed_set:
            continue

        if sp500_only and sym not in SP500_MEMBERS:
            continue

        if sym in merged_addition_dates:
            start = merged_addition_dates[sym]
        else:
            _missing_dates.append(sym)
            if strict:
                continue
            else:
                start = UNIVERSE_START

        records.append({
            "symbol":     sym,
            "sector":     sector_map.get(sym) or "Unknown",
            "start_date": start,
            "end_date":   UNIVERSE_END,
            "status":     "active",
        })

    if _missing_dates:
        mode_label = "EXCLUDED (strict mode)" if strict else f"defaulted to {UNIVERSE_START}"
        detail = f": {sorted(_missing_dates)}" if strict else ""
        print(
            f"[Universe] {len(_missing_dates)} symbols have no recorded addition date "
            f"— {mode_label}{detail}"
        )

    for sym, sector, removed, reason in REMOVED_STOCKS:
        if sym in current_symbols:
            continue
        start = SYMBOL_ADDITION_DATES.get(sym, UNIVERSE_START)
        records.append({
            "symbol":     sym,
            "sector":     sector_map.get(sym) or sector,
            "start_date": start,
            "end_date":   removed,
            "status":     "removed",
            "reason":     reason,
        })

    df = pd.DataFrame(records)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"]   = pd.to_datetime(df["end_date"])
    return df


def get_universe_on_date(date, universe_df=None):
    if universe_df is None:
        universe_df = build_historical_universe()
    if isinstance(date, str):
        date = pd.Timestamp(date)
    mask = (universe_df["start_date"] <= date) & (universe_df["end_date"] >= date)
    return sorted(universe_df.loc[mask, "symbol"].tolist())


def get_all_historical_symbols(universe_df=None):
    if universe_df is None:
        universe_df = build_historical_universe()
    return sorted(universe_df["symbol"].unique().tolist())


def get_removed_symbols():
    records = []
    for sym, sector, removed, reason in REMOVED_STOCKS:
        records.append({
            "symbol":       sym,
            "sector":       sector,
            "removed_date": removed,
            "reason":       reason,
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    universe = build_historical_universe(sp500_only=True)
    print("=" * 60)
    print("  HISTORICAL UNIVERSE (sp500_only=True, Option B)")
    print("=" * 60)
    print("  Active stocks:  %d" % len(universe[universe["status"] == "active"]))
    print("  Removed stocks: %d" % len(universe[universe["status"] == "removed"]))
    print("  Total universe: %d" % len(universe))
    print("")
    for test_date in ["2020-01-15", "2021-06-15", "2022-06-15", "2023-06-15", "2024-06-15", "2025-06-15"]:
        syms = get_universe_on_date(test_date, universe)
        print("  %s: %d stocks" % (test_date, len(syms)))
    print("")
