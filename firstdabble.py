import finsymbols
import pandas as pd
from pandas_datareader import DataReader


symbols_list = ['AAPL', 'TSLA', 'YHOO','GOOG', 'MSFT','GILD']
symbols = []

for ticker in symbols_list:
    r = DataReader(ticker, "yahoo", '2017-01-01')
    r['Symbol'] = ticker
    symbols.append(r)
df = pd.concat(symbols)
print df
