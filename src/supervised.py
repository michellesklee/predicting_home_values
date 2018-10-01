import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import accuracy_score, recall_score, precision_score
from regression import standardize_data, rmse
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
import matplotlib
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

data = pd.read_csv('/Users/michellelee/galvanize/week4/predicting_home_values/data/main.csv')
df = data.copy()

col_2016 = [col for col in df if col.startswith('2016')]
col_2017 = [col for col in df if col.startswith('2017')]
col_2018 = [col for col in df if col.startswith('2018')]

df_2016 = df[col_2016]
df_2017 = df[col_2017]
df_2018 = df[col_2018]

df_2016.drop('2016_Zip_Zhvi_AllHomes', inplace = True, axis = 1)
df_2017.drop('2017_Zip_Zhvi_AllHomes', inplace = True, axis = 1)
df_2018.drop('2018_Zip_Zhvi_AllHomes', inplace = True, axis = 1)

X_2016 = df_2016.values
X_2017 = df_2017.values
X_2018 = df_2018.values

y_2016 = df['2016_Zip_Zhvi_AllHomes'].values
y_2017 = df['2017_Zip_Zhvi_AllHomes'].values
y_2018 = df['2018_Zip_Zhvi_AllHomes'].values


X_train_std, X_test_std, y_train, y_test = standardize_data(X_2016, y_2016)

# Decision Tree
dt = DecisionTreeRegressor()
dt.fit(X_train_std, y_train)
dt_preds = dt.predict(X_test_std)

print('Decision Tree RMSE: {}'.format(rmse(dt_preds, y_test)))

# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train_std, y_train)
rf_preds = rf.predict(X_test_std)

print('Random Forest RMSE: {}'.format(rmse(rf_preds, y_test)))

# Bagging
bag = BaggingRegressor()
bag.fit(X_train_std, y_train)
bag_preds = bag.predict(X_test_std)

print('Bagging RMSE: {}'.format(rmse(bag_preds, y_test)))

# K Nearest Neighbors
knn = KNeighborsRegressor()
knn.fit(X_train_std, y_train)
knn_preds = knn.predict(X_test_std)

print('KNN RMSE: {}'.format(rmse(knn_preds, y_test)))


# Gradient Boosting
boost = GradientBoostingRegressor()
boost.fit(X_train_std, y_train)
boost_preds = boost.predict(X_test_std)

print('Gradient Boosting RMSE: {}'.format(rmse(boost_preds, y_test)))

# Adaboost
ada = AdaBoostRegressor()
ada.fit(X_train_std, y_train)
ada_preds = ada.predict(X_test_std)

print('AdaBoost RMSE: {}'.format(rmse(ada_preds, y_test)))
