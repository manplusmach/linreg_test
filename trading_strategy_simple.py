# trading strategy for a simple test

import numpy as np
import pandas as pd
import pandas.io.data as web

sp500 = web.DataReader('^GSPC', data_source = 'yahoo', start = '1/1/2000', end = '12/31/2014')

sp500['42d'] = np.round(pd.rolling_mean(sp500['Close'], window = 100), 2)
sp500['252d'] = np.round(pd.rolling_mean(sp500['Close'], window = 512), 2)

sp500[['Close', '42d', '252d']].plot(grid = True, figsize = (8,5))

sp500['42-252'] = sp500['42d'] - sp500['252d']

SD = 50

sp500['Regime'] = np.where(sp500['42-252'] > SD, 1, 0)
sp500['Regime'] = np.where(sp500['42-252'] < -SD, -1, sp500['Regime'])
sp500['Regime'].value_counts()

sp500['Market'] = np.log(sp500['Close'] / sp500['Close'].shift(1))

sp500['Strategy'] = sp500['Regime'].shift(1) * sp500['Market']

sp500[['Market', 'Strategy']].cumsum().apply(np.exp).plot(grid=True,figsize = (8,5))

print SD