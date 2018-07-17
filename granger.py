import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR
from sklearn import linear_model, datasets
from datetime import datetime

def ADFtest(v, crit='5%', max_d=6, reg='nc', autolag='AIC'):
    """ Augmented Dickey Fuller test
        Inputs: v - ndarray matrix
        Outputs: boolean - true if v pass the test
    """

    boolean = False
    adf = sm.tsa.stattools.adfuller(v, max_d, reg, autolag)
    # print(adf)
    if(adf[0] < adf[4][crit]):
        pass
    else:
        boolean = True

    return boolean

def granger_causes(data, maxlag, ic=None):
    """ Check whether one time series G-causes the other
        Model:
               y_t = Sum_{j=1}^p A_{1, j} x_{t-j} + Sum_{j=1}^p A_{2, j} y_{t-j} + Error
        Inputs: data: T x 2
        Output: F-statistic
    """
    model = VAR(data[["diff_ClosePrice","diff_Signal"]])
    results = model.fit(maxlag, ic=ic)
    results.test_causality("diff_ClosePrice","diff_Signal", kind='f', signif=0.05)
    return model, results

def outliers_iqr(x):
    """ Outlier detection
        Inputs: x - ndarray matrix
        Outputs: boolean - true if v pass the test
    """
    quartile_1, quartile_3 = np.percentile(x, [25, 75])
    # interquartile range
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((x > upper_bound) | (x < lower_bound))

# Import the data
# Note that the data file should be in the folder with the Python file
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
data = pd.read_excel('ResearchDatasetV2.0.xlsx', index_col = 'Date', columns = ['Signal','ClosePrice'], parse_dates=['Date'], date_parser=dateparse)

# A first look at the data
print(data.head())

# Plot the Signal, ClosePrice vs Time
fig, ax1 = plt.subplots(figsize = (6,4))
ax1.plot(data['ClosePrice'], 'b-', label = 'ClosePrice')
# ax1.xticks(rotation=45)
ax1.set_ylabel("Price ($)")
ax1_s = ax1.twinx()
ax1_s.plot(data['Signal'], 'r-', label = 'Signal')
ax1_s.set_ylabel("Signal")
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_s.get_legend_handles_labels()
ax1_s.legend(lines + lines2, labels + labels2, loc=0)
plt.savefig('Signal, ClosePrice vs Time.png',bbox_inches='tight', dpi=300)
plt.close(fig)

# Plot Signal vs ClosePrice
fig2, ax2 = plt.subplots(figsize = (6,4))
ax2.loglog(data['Signal'], data['ClosePrice'], 'k.')
ax2.set_ylabel("Price ($)")
ax2.set_xlabel("Signal")
plt.savefig('Signal vs ClosePrice.png',bbox_inches='tight', dpi=300)
plt.close(fig2)

# A first look at the data shows that Signal and ClosePrice have a strong linear relationship, with a small subset of outliers
# Robustly fit linear model with TheilSen algorithm
X = data['Signal']
X = sm.add_constant(X)
y = data['ClosePrice'].to_frame()

# ThielSen fitting
theilsen = linear_model.TheilSenRegressor()
theilsen.fit(X, y)

#  Estimated coefficients
print("TheilSen Estimated coefficients:")
print(theilsen.coef_)

# Now try to find the outliers
# We use a boxplot method
fig3, ax3 = plt.subplots(figsize = (6,4))
ax3.boxplot([data['Signal'], data['ClosePrice']])
# plt.xticks(["Signal", "Price ($)"])
plt.savefig('Signal, ClosePrice - Boxplot.png',bbox_inches='tight', dpi=300)
plt.close(fig3)

# Find outliers
# Outliers are defined as quartile1-iqr or quantile3+iqr
outliers_Signal = outliers_iqr(data['Signal'])
outliers_ClosePrice = outliers_iqr(data['ClosePrice'])

print("Outliers are located:")
print(outliers_Signal)
print(outliers_ClosePrice)

# Adjust the outliers using TheilSen parameters
for outlier1 in np.nditer(outliers_Signal):
    data['Signal'].iloc[int(outlier1)] = (data['ClosePrice'].iloc[int(outlier1)] - theilsen.coef_[0])/theilsen.coef_[1]

for outlier2 in np.nditer(outliers_ClosePrice):
    data['ClosePrice'].iloc[int(outlier2)] = theilsen.coef_[0] + data['Signal'].iloc[int(outlier2)] * theilsen.coef_[1]

# Check the adjusted data
# Plot Signal vs ClosePrice - Adjusted
fig4, ax4 = plt.subplots(figsize = (6,4))
ax4.plot(data['Signal'], data['ClosePrice'], 'k.')
ax4.set_ylabel("Price ($)")
ax4.set_xlabel("Signal")
plt.savefig('Signal vs ClosePrice - Adjusted.png',bbox_inches='tight', dpi=300)
plt.close(fig4)

# Plot Signal, ClosePrice vs Time - Adjusted
fig5, ax5 = plt.subplots(figsize = (6,4))
ax5.plot(data['ClosePrice'], 'b-', label = 'ClosePrice')
ax5.set_ylabel("Price ($)")
ax5_s = ax5.twinx()
ax5_s.plot(data['Signal'], 'r-', label = 'Signal')
ax5_s.set_ylabel("Signal")
lines, labels = ax5.get_legend_handles_labels()
lines2, labels2 = ax5_s.get_legend_handles_labels()
ax5_s.legend(lines + lines2, labels + labels2, loc=0)
plt.legend()
plt.savefig('Signal, ClosePrice vs Time - Adjusted.png',bbox_inches='tight', dpi=300)
plt.close(fig5)

# Check if the data is stationary
print(ADFtest(data['Signal']))
print(ADFtest(data['ClosePrice']))

# Take differentials
data['diff_Signal']=data['Signal'].diff()
data['diff_ClosePrice']=data['ClosePrice'].diff()
data.dropna(inplace=True)

# Check if the data is stationary
print(ADFtest(data['diff_Signal']))
print(ADFtest(data['diff_ClosePrice']))

# Granger causality test
granger_test_result = ts.grangercausalitytests(data[["diff_ClosePrice","diff_Signal"]], maxlag=10, verbose=False)

print("Granger Causality test results")

for key in granger_test_result.keys():
    F_test = granger_test_result[key][0]['params_ftest'][0]
    P_value = granger_test_result[key][0]['params_ftest'][1]
    print(key, F_test, P_value)

# Another Granger causality test
model, results = granger_causes(data[["diff_ClosePrice","diff_Signal"]], maxlag=10, ic='bic')
results.summary()