# imports
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy as np
import scipy, scipy.stats

data = pd.read_csv('Advertising.csv', index_col=None, usecols=None)
data.head()
data.shape

# visualize the relationship between the features and the response using scatterplots
sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.7)

# create a fitted model with all three features
lm1 = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()

# print the coefficients
lm1.params

# create X and y
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data.Sales

# instantiate and fit
lm2 = LinearRegression()
lm2.fit(X, y)
print(lm2.intercept_)
print(lm2.coef_)

result = sm.OLS( Y, X ).fit()
result.summary()

# only include TV and Radio in the model

# instantiate and fit model
lm1 = smf.ols(formula='Sales ~ TV + Radio', data=data).fit()

# calculate r-square
lm1.rsquared

# add Newspaper to the model (which we believe has no association with Sales)
lm1 = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()
lm1.rsquared

# include Newspaper
X = data[['TV', 'Radio', 'Newspaper']]
y = data.Sales

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Instantiate model
lm2 = LinearRegression()

# Fit Model
lm2.fit(X_train, y_train)

# Predict
y_pred = lm2.predict(X_test)

# RMSE
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# exclude Newspaper
X = data[['TV', 'Radio']]
y = data.Sales

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Instantiate model
lm2 = LinearRegression()

# Fit model
lm2.fit(X_train, y_train)

# Predict
y_pred = lm2.predict(X_test)

# RMSE
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# dummies
# create a new Series called Size_large
data['Size_large'] = data.Size.map({'small':0, 'large':1})
data.head()

# create X and y
feature_cols = ['TV', 'Radio', 'Newspaper', 'Size_large']
X = data[feature_cols]
y = data.Sales

# instantiate
lm2 = LinearRegression()
# fit
lm2.fit(X, y)

# print coefficients
list(zip(feature_cols, lm2.coef_))

# concatenate the dummy variable columns onto the DataFrame (axis=0 means rows, axis=1 means columns)
data = pd.concat([data, area_dummies], axis=1)
data.head()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X_label_encoder = LabelEncoder()
X[:,0] = X_label_encoder.fit_transform(X[:,0])
one_hot_encoder = OneHotEncoder(categorical_features=[0])
X = one_hot_encoder.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# Training our model# Trainin
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting values using our trained model
y_pred = regressor.predict(X_test)

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
ex_var_score = explained_variance_score(y_test, y_pred)
m_absolute_error = mean_absolute_error(y_test, y_pred)
m_squared_error = mean_squared_error(y_test, y_pred)
r_2_score = r2_score(y_test, y_pred)

import statsmodels.api as sm # import statsmodels

X = df["RM"] ## X usually means our input variables (or independent variables)
y = target["MEDV"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
model.summary()

df1 = pd.read_csv(r"E:\Business\Economic Indicators\Consumer Price Index - Core (YoY) - European Monetary Union.csv",
                  index_col=[0], parse_dates=[0])
df2 = pd.read_csv(r"E:\Business\Economic Indicators\Private loans (YoY) - European Monetary Union.csv",
                  index_col=[0], parse_dates=[0])
df3 = pd.read_csv(r"E:\Business\Economic Indicators\Current Account s.a - European Monetary Union.csv",
                  index_col=[0], parse_dates=[0])

finaldf = pd.concat([df1, df2, df3], axis=1, join='inner').sort_index()

import os

os.chdir('E:\\Business\\Economic Indicators')

dfs = [pd.read_csv(f, index_col=[0], parse_dates=[0])
        for f in os.listdir(os.getcwd()) if f.endswith('csv')]

finaldf = pd.concat(dfs, axis=1, join='inner').sort_index()

import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())

# https://www.datascience.com/blog/stock-price-time-series-arima