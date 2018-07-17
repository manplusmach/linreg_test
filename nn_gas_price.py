from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt


natgas = pd.read_csv('HDD_CDD_Storage.csv')


AllData = natgas[['HDD_New England','HDD_Middle Atlantic','HDD_EN Central','HDD_WN Central','HDD_South Atlantic','HDD_ES Central','HDD_WS Central',
                  'HDD_Mountain','HDD_Pacific',	'CDD_New England','CDD_Middle Atlantic','CDD_EN Central','CDD_WN Central','CDD_South Atlantic',
                  'CDD_ES Central','CDD_WS Central','CDD_Mountain','CDD_Pacific','Stor_East','Stor_Midwest','Stor_Mountain','Stor_Pacific',
                  'Stor_South Central','Stor_Salt','Stor_NonSalt','Stor_Total']]

AllData['Stor_Total_lag1'] = AllData['Stor_Total'].shift()
# AllData['Stor_Total_lag2'] = AllData['Stor_Total'].shift(2)
AllData['Stor_East_lag1'] = AllData['Stor_East'].shift()

# AllData = AllData.astype(float)
# pd.to_numeric(X_all, errors='ignore')
# X_all['HDD'] = np.reciprocal(X_all['HDD'].astype(np.float32))
# X_all['CDD'] = np.reciprocal(X_all['CDD'].astype(np.float32))
# X_all = X_all.drop(X_all[X_all['HDD'] > 0.2].index)
# X_all = X_all.drop(X_all[X_all['CDD'] > 0.2].index)
# X_all = X_all.drop(X_all[X_all['HDD'] < -10].index)
# X_all = X_all.drop(X_all[X_all['CDD'] < -10].index)
# X_all.replace([np.inf, -np.inf], np.nan, inplace = True)
# X_all.dropna(inplace = True)

X_train= AllData[['HDD_New England','HDD_Middle Atlantic','HDD_EN Central','HDD_WN Central','HDD_South Atlantic','HDD_ES Central','HDD_WS Central',
                  'HDD_Mountain','HDD_Pacific',	'CDD_New England','CDD_Middle Atlantic','CDD_EN Central','CDD_WN Central','CDD_South Atlantic',
                  'CDD_ES Central','CDD_WS Central','CDD_Mountain','CDD_Pacific','Stor_Total_lag1']][1:250]

y_train = AllData['Stor_Total'][1:250]

X_test = AllData[['HDD_New England','HDD_Middle Atlantic','HDD_EN Central','HDD_WN Central','HDD_South Atlantic','HDD_ES Central','HDD_WS Central',
                  'HDD_Mountain','HDD_Pacific',	'CDD_New England','CDD_Middle Atlantic','CDD_EN Central','CDD_WN Central','CDD_South Atlantic',
                  'CDD_ES Central','CDD_WS Central','CDD_Mountain','CDD_Pacific','Stor_Total_lag1']][250:]
y_test = AllData['Stor_Total'][250:]

# cancer = load_breast_cancer()
#
# cancer.keys()
#
# cancer['data'].shape
#
# X = cancer['data']
# y = cancer['target']

# X_train,X_test,y_train,y_test = train_test_split(X,y)

# scaler = StandardScaler()
#
# scaler.fit(X_train)
#
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

mlp = MLPRegressor(hidden_layer_sizes=(19,19),activation='relu')

mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

print 'Score is'
print mlp.score(X_test, y_test)

plt.scatter(y_test,predictions)

plt.xlim([0, 4500])
plt.ylim([0, 4500])

# print(confusion_matrix(y_test, predictions))
#
# print(classification_report(y_test,predictions))

len(mlp.coefs_)



