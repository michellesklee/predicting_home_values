from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from math import sqrt

import matplotlib
matplotlib.rc('figure', figsize = (12, 12))
matplotlib.rc('font', size = 14)
matplotlib.rc('axes.spines', top = False, right = False)
matplotlib.rc('axes', grid = False)
matplotlib.rc('axes', facecolor = 'white')

data = pd.read_csv('~/galvanize/week4/analytic_capstone/main.csv')
df = data.copy()

#create dataframes for each year
col_2016 = [col for col in df if col.startswith('2016')]
col_2017 = [col for col in df if col.startswith('2017')]
col_2018 = [col for col in df if col.startswith('2018')]

df_2016 = df[col_2016]
df_2017 = df[col_2017]
df_2018 = df[col_2018]

#dropped redundant and highly correlated features
df_2016.drop(['2016_Zip_Zhvi_AllHomes', '2016_Zip_Zhvi_BottomTier',
        '2016_Zip_Zhvi_MiddleTier', '2016_Zip_Zhvi_TopTier',
        '2016_Zip_MedianListingPricePerSqft_AllHomes',
        '2016_Zip_MedianRentalPricePerSqft_AllHomes',
        '2016_Zip_MedianPctOfPriceReduction_AllHomes',
        '2016_Zip_MedianValuePerSqft_AllHomes',
        '2016_Zip_PctOfListingsWithPriceReductions_AllHomes',
        '2016_Zip_Zri_AllHomesPlusMultifamily',
        '2016_Zip_Zri_AllHomes',
        '2016_Zip_ZriPerSqft_AllHomes'
        '2016_Valet_Location_License',
        '2016_Grand_Total'], inplace = True, axis = 1)
df_2017.drop(['2017_Zip_Zhvi_AllHomes', '2017_Zip_Zhvi_BottomTier',
        '2017_Zip_Zhvi_MiddleTier', '2017_Zip_Zhvi_TopTier',
        '2017_Zip_MedianRentalPricePerSqft_AllHomes',
        '2017_Zip_MedianPctOfPriceReduction_AllHomes',
        '2017_Zip_MedianListingPricePerSqft_AllHomes',
        '2017_Zip_MedianValuePerSqft_AllHomes',
        '2017_Zip_PctOfListingsWithPriceReductions_AllHomes',
        '2017_Zip_Zri_AllHomesPlusMultifamily',
        '2017_Zip_Zri_AllHomes',
        '2017_Zip_ZriPerSqft_AllHomes'
        '2017_Swimming_Pool','2017_Valet_Location_License',
        '2017_Grand_Total'], inplace = True, axis = 1)
df_2018.drop(['2018_Zip_Zhvi_AllHomes', '2018_Zip_Zhvi_BottomTier',
        '2018_Zip_Zhvi_MiddleTier', '2018_Zip_Zhvi_TopTier',
        '2018_Zip_MedianListingPricePerSqft_AllHomes',
        '2018_Zip_MedianRentalPricePerSqft_AllHomes',
        '2018_Zip_MedianPctOfPriceReduction_AllHomes',
        '2018_Zip_MedianValuePerSqft_AllHomes',
        '2018_Zip_PctOfListingsWithPriceReductions_AllHomes',
        '2018_Zip_Zri_AllHomesPlusMultifamily',
        '2018_Zip_Zri_AllHomes',
        '2018_Zip_ZriPerSqft_AllHomes'], inplace = True, axis = 1)


X_2016 = df_2016.values
X_2017 = df_2017.values
X_2018 = df_2018.values

y_2016 = df['2016_Zip_Zhvi_AllHomes'].values
y_2017 = df['2017_Zip_Zhvi_AllHomes'].values
y_2018 = df['2018_Zip_Zhvi_AllHomes'].values



#rmse
def rmse(true, predicted):
    return np.sqrt(np.mean((true - predicted) ** 2))

#split data
X_train_2016, X_test_2016, y_train_2016, y_test_2016 = train_test_split(X_2016, y_2016)
scaler = StandardScaler()
scaler.fit(X_train_2016)
X_train_std_2016 = scaler.transform(X_train_2016)
X_test_std_2016 = scaler.transform(X_test_2016)


#1a. Linear regression with train and test on 2016
linear = LinearRegression()
linear.fit(X_train_std_2016, y_train_2016)

train_predicted_2016 = linear.predict(X_train_std_2016)
test_predicted_2016 = linear.predict(X_test_std_2016)
rmse_train_2016_2016 = rmse(y_train_2016, train_predicted_2016)
rmse_test_2016_2016 = rmse(y_test_2016, test_predicted_2016)

#plot true vs. predicted 2016
plt.scatter(y_test_2016, test_predicted_2016, alpha = .75, c="#7B241C")
line = np.linspace(200000, 550000, 1000)
plt.plot(line,line)
plt.title('Linear Model 2016-2016')
plt.xlabel('True 2016 Home Values')
plt.ylabel('Predicted 2016 Home Values')
#plt.savefig('linear-2016-2016.png')
plt.show()

#1b. Linear regression predicting on 2017
X_2017_std = scaler.transform(X_2017)
predicted_2017 = linear.predict(X_2017_std)
rmse_test_2017 = rmse(y_2017, predicted_2017)

#plot true vs. predicted 2016
plt.scatter(y_2017, predicted_2017)
line = np.linspace(200000, 700000, 1000)
plt.plot(line,line)
plt.title('Linear Model 2016-2017')
plt.xlabel('True 2016 Home Values')
plt.ylabel('Predicted 2017 Home Values')
plt.savefig('linear-2016-2017.png')
plt.show()


#2a. RidgeCV 2016-2016
ridge = RidgeCV(alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1,
        0.3, 0.6, 1], normalize=True, cv=10)
ridge.fit(X_train_std_2016, y_train_2016)

test_predicted_ridge_2016 = ridge.predict(X_test_std_2016)
rmse_test_ridge_2016 = rmse(y_test_2016, test_predicted_ridge_2016)

plt.scatter(y_test_2016, test_predicted_ridge_2016)
line = np.linspace(200000, 550000, 1000)
plt.plot(line,line)
plt.title('Ridge Model 2016-2016')
plt.xlabel('True 2016 Home Values')
plt.ylabel('Predicted 2016 Home Values')
plt.savefig('ridge-2016-2016.png')
plt.show()

#2b. RidgeCV 2016-2017
predicted_ridge_2017 = ridge.predict(X_2017_std)
rmse_test_ridge_2017 = rmse(y_2017, predicted_ridge_2017)

plt.scatter(y_2017, predicted_ridge_2017)
line = np.linspace(200000, 700000, 1000)
plt.plot(line,line)
plt.title('Ridge Model 2016-2017')
plt.xlabel('True 2016 Home Values')
plt.ylabel('Predicted 2017 Home Values')
plt.savefig('ridge-2016-2017.png')
plt.show()

#3a. Lasso 2016-2016
lasso = Lasso(alpha=.8, normalize=True, max_iter=10000, selection='random')
lasso.fit(X_train_std_2016, y_train_2016)
test_predicted_lasso_2016 = lasso.predict(X_test_std_2016)
rmse_test_lasso_2016 = rmse(y_test_2016, test_predicted_lasso_2016) # 17951.417123492214 @ alpha=.5

plt.scatter(y_test_2016, test_predicted_lasso_2016)
line = np.linspace(200000, 550000, 1000)
plt.plot(line,line)
plt.title('Lasso Model 2016-2016')
plt.xlabel('True 2016 Home Values')
plt.ylabel('Predicted 2016 Home Values')
plt.savefig('lasso-2016-2016.png')
plt.show()

#3b. Lasso 2016-2017
predicted_lasso_2017 = lasso.predict(X_2017_std)
rmse_test_lasso_2017 = rmse(y_2017, predicted_lasso_2017) #49549.69139393983 @ alpha =.5

plt.scatter(y_2017, predicted_lasso_2017)
line = np.linspace(200000, 700000, 1000)
plt.plot(line,line)
plt.title('Lasso Model 2016-2017')
plt.xlabel('True 2016 Home Values')
plt.ylabel('Predicted 2017 Home Values')
plt.savefig('lasso-2016-2017.png')
plt.show()
