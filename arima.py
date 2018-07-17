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

def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
data = pd.read_excel('data.xlsx', index_col = 'Date', columns = ['Signal','ClosePrice'], parse_dates=['Date'], date_parser=dateparse)

print(data.head())

data.head(10)
data.describe()
fig, ax1 = plt.subplots(figsize = (6,4))
ax1.plot(data['ClosePrice'], 'b-', label = 'ClosePrice')
# ax1.xticks(rotation=45)
ax1.set_ylabel("Price ($)")
# ax1.set_xlabel("Date")
ax2 = ax1.twinx()
ax2.plot(data['Signal'], 'r-', label = 'Legend')
ax2.set_ylabel("Signal")
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
# plt.legend()
# plt.show()
plt.savefig('First Look.png',bbox_inches='tight', dpi=300)

fig2, ax3 = plt.subplots(figsize = (6,4))
ax3.loglog(data['Signal'], data['ClosePrice'], 'k.')
plt.savefig('First Look - Signal vs ClosePrice.png',bbox_inches='tight', dpi=300)

# A first look at the data shows that Signal and ClosePrice have a strong linear relationship, with a small subset of outliers
# Robustly fit linear model with TheilSen algorithm
X = data['Signal']
X = sm.add_constant(X)
y = data['ClosePrice'].to_frame()

theilsen = linear_model.TheilSenRegressor()
theilsen.fit(X, y)

# Compare estimated coefficients
print("TheilSen Estimated coefficients:")
print(theilsen.coef_)

# Now try to find the outliers
# We use a boxplot method
fig3, ax4 = plt.subplots(figsize = (6,4))
plt.boxplot([data['Signal'], data['ClosePrice']])
ax4.set_xlabel(["Signal","ClosePrice"])
plt.savefig('First Look - Boxplot.png',bbox_inches='tight', dpi=300)

# Adjust the outliers
# mean = numpy.mean(elements, axis=0)
# sd = numpy.std(elements, axis=0)
#
# final_list = [x for x in arr if (x > mean - 2 * sd)]
# final_list = [x for x in final_list if (x < mean + 2 * sd)]
outliers_Signal = outliers_iqr(data['Signal'])
outliers_ClosePrice = outliers_iqr(data['ClosePrice'])

print(outliers_Signal)
print(outliers_ClosePrice)

for outlier1 in np.nditer(outliers_Signal):
    print(outlier1)
    print(theilsen.coef_[0])
    print(data['Signal'].loc[outlier1])
    print(data['ClosePrice'].loc[outlier1])
    data['Signal'].loc[outlier1] = (data['ClosePrice'].loc[outlier1] - theilsen.coef_[0])/theilsen.coef_[1]

for outlier2 in np.nditer(outliers_ClosePrice):
    data['ClosePrice'].loc[outlier2] = theilsen.coef_[0] + data['Signal'].loc[outlier2] * theilsen.coef_[1]

fig4, ax5 = plt.subplots(figsize = (6,4))
ax5.loglog(data['Signal'], data['ClosePrice'], 'k.')
plt.savefig('First Look - Signal vs ClosePrice2.png',bbox_inches='tight', dpi=300)

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
