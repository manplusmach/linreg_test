import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
# Import the Time Series library
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model, datasets
# Import Datetime and the Pandas DataReader
from datetime import datetime

# def hurst(ts):
# 	"""Returns the Hurst Exponent of the time series vector ts"""
# 	# Create the range of lag values
# 	lags = range(2, 100)
#
# 	# Calculate the array of the variances of the lagged differences
# 	tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
#
# 	# Use a linear fit to estimate the Hurst Exponent
# 	poly = polyfit(log(lags), log(tau), 1)
#
# 	# Return the Hurst exponent from the polyfit output
# 	return poly[0]*2.0

# Create a Gometric Brownian Motion, Mean-Reverting and Trending Series
# gbm = log(cumsum(randn(100000))+1000)
# mr = log(randn(100000)+1000)
# tr = log(cumsum(randn(100000)+1)+1000)

# Output the Hurst Exponent for each of the above series
# and the price of Google (the Adjusted Close price) for
# the ADF test given above in the article
# print "Hurst(GBM):   %s" % hurst(gbm)
# print "Hurst(MR):    %s" % hurst(mr)
# print "Hurst(TR):    %s" % hurst(tr)
def normalize(x):

    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x.values)
    x_norm = pd.DataFrame(x_norm, index=x.index, columns=x.columns)

    return x_norm

def scores(models, X, y):

    for model in models:
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_pred)
        print("Accuracy Score: {0:0.2f} %".format(acc * 100))
        print("F1 Score: {0:0.4f}".format(f1))
        print("Area Under ROC Curve Score: {0:0.4f}".format(auc))
# Output the results of the Augmented Dickey-Fuller test for Google
# with a lag order value of 1
# ts.adfuller(goog['Adj Close'], 1)

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
data = pd.read_excel('data - Copy.xlsx', index_col = 'Date', columns = ['Signal','ClosePrice'], parse_dates=['Date'], date_parser=dateparse)

# X = data['Signal']
# X = sm.add_constant(X)
# y = data['ClosePrice'].to_frame()
#
# # Robustly fit linear model with RANSAC algorithm
# ransac = linear_model.RANSACRegressor()
# ransac.fit(X, y)
# inlier_mask = ransac.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)
#
# # Predict data of estimated models
# line_y_ransac = ransac.predict(X)
#
# # Compare estimated coefficients
# print("Estimated coefficients (true, linear regression, RANSAC):")
# print(ransac.estimator_.coef_)

X = data['Signal']
X = sm.add_constant(X)
y = data['ClosePrice'].to_frame()

# Robustly fit linear model with RANSAC algorithm
theilsen = linear_model.TheilSenRegressor()
theilsen.fit(X, y)
# inlier_mask = ransac.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated models
# line_y_ransac = ransac.predict(X)

# Compare estimated coefficients
print("Estimated coefficients (true, linear regression, RANSAC):")
print(theilsen.coef_)



print(data.head())

data.tail(10)
data.describe()
fig, ax1 = plt.subplots(figsize = (6,4))
ax1.plot(data['ClosePrice'], color = 'b', label = 'ClosePrice')
# ax1.xticks(rotation=45)
ax1.set_ylabel("Price ($)")
# ax1.set_xlabel("Date")
# ax1.xaxis.set_major_locator(plt.MaxNLocator(7))
ax2 = ax1.twinx()
ax2.plot(data['Signal'], color = 'r', label = 'Legend')
ax2.set_ylabel("Signal")
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
# ax2.xaxis.set_major_locator(plt.MaxNLocator(7))
# plt.legend()
plt.show()
plt.savefig('First Look.png',bbox_inches='tight', dpi=300)


print(sm.tsa.stattools.adfuller(data['Signal']))
print(sm.tsa.stattools.adfuller(data['ClosePrice']))

# naive regression
data['const']=1
model1=sm.OLS(endog=data['ClosePrice'],exog=data[['Signal','const']])
results1=model1.fit()
print(results1.summary())

data['diff_Signal']=data['Signal'].diff()
data['diff_Signal_lag']=data['diff_Signal'].shift()

data['diff_ClosePrice']=data['ClosePrice'].diff()
data.dropna(inplace=True)
model3=sm.tsa.ARIMA(endog=data['diff_ClosePrice'],exog=data['diff_Signal_lag'],order=[1,1,0])
results3=model3.fit()
print(results3.summary())


# df[['Marketing','Sales']].plot()
# plt.show()

# data_train = data['2011-01-01':'2017-01-01']
# data_train = rebalance(data_train)
# y = data_train.target
# X = data_train.drop('target', axis=1)
# X = normalize(X)
# data_val = data['2017-01-01':]
# y_val = data_val.target
# X_val = data_val.drop('target', axis=1)
# X_val = normalize(X_val)
#
