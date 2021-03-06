from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import style
from math import sqrt
from utils import XyScaler
from sklearn.base import clone
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from regression import rmse


style.use('ggplot')
import matplotlib
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

data = pd.read_csv('/Users/michellelee/galvanize/week4/analytic_capstone/data/main.csv')
df = data.copy()

#------------------ CREATE DATAFRAMES FOR EACH YEAR -------------------------------

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


#------------------ LINEAR REGRESSION -------------------------------

X_train_2016, X_test_2016, y_train_2016, y_test_2016 = train_test_split(X_2016, y_2016)
scaler = StandardScaler()
scaler.fit(X_train_2016)
X_train_std_2016 = scaler.transform(X_train_2016)
X_test_std_2016 = scaler.transform(X_test_2016)


def cv(X, y, base_estimator, n_folds, random_seed=150):
    kf = KFold(n_splits=n_folds, random_state=random_seed)
    test_cv_errors, train_cv_errors = np.empty(n_folds), np.empty(n_folds)
    for idx, (train, test) in enumerate(kf.split(X_train_2016)):
        # Split into train and test
        X_cv_train, y_cv_train = X_2016[train], y_2016[train]
        X_cv_test, y_cv_test = X_2016[test], y_2016[test]
        # Standardize data
        standardizer = XyScaler()
        standardizer.fit(X_cv_train, y_cv_train)
        X_cv_train_std, y_cv_train_std = standardizer.transform(X_cv_train, y_cv_train)
        X_cv_test_std, y_cv_test_std = standardizer.transform(X_cv_test, y_cv_test)
        # Fit estimator
        estimator = clone(base_estimator)
        estimator.fit(X_cv_train_std, y_cv_train_std)
        # Measure performance
        y_hat_train = estimator.predict(X_cv_train_std)
        y_hat_test = estimator.predict(X_cv_test_std)
        # Calclate the error metrics
        train_cv_errors[idx] = rmse(y_cv_train_std, y_hat_train)
        test_cv_errors[idx] = rmse(y_cv_test_std, y_hat_test)
    return train_cv_errors, test_cv_errors

def train_at_various_alphas(X, y, model, alphas, n_folds=10, **kwargs):
    cv_errors_train = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                     columns=alphas)
    cv_errors_test = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                        columns=alphas)
    for alpha in alphas:
        train_fold_errors, test_fold_errors = cv(X, y, model(alpha=alpha, **kwargs), n_folds=n_folds)
        cv_errors_train.loc[:, alpha] = train_fold_errors
        cv_errors_test.loc[:, alpha] = test_fold_errors
    return cv_errors_train, cv_errors_test

def get_optimal_alpha(mean_cv_errors_test):
    alphas = mean_cv_errors_test.index
    optimal_idx = np.argmin(mean_cv_errors_test.values)
    optimal_alpha = alphas[optimal_idx]
    return optimal_alpha

#------------------ LINEAR REGRESSION -------------------------------

#1a. Linear regression with train and test on 2016
linear = LinearRegression()
linear.fit(X_train_std_2016, y_train_2016)

train_predicted_2016 = linear.predict(X_train_std_2016)
test_predicted_2016 = linear.predict(X_test_std_2016)
rmse_train_2016_2016 = rmse(y_train_2016, train_predicted_2016)
rmse_test_2016_2016 = rmse(y_test_2016, test_predicted_2016)

#1b. Linear regression predicting on 2017
X_2017_std = scaler.transform(X_2017)
predicted_2017 = linear.predict(X_2017_std)
rmse_test_2017 = rmse(y_2017, predicted_2017)

#plot true vs. predicted
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(28, 8))
ax0.scatter(y_test_2016, test_predicted_2016, alpha = .75, color='#229954')
line = np.linspace(150000, 700000, 1000)
ax0.plot(line,line, color='grey')
ax0.set_title('Linear Model 2016-2016')
ax0.set_xlabel('True 2016 Home Values')
ax0.set_ylabel('Predicted 2016 Home Values')
ax0.set_xlim(150000, 700000)
ax0.set_ylim(150000, 700000)

ax1.scatter(y_2017, predicted_2017, alpha = .75, color='#229954')
line = np.linspace(150000, 700000, 1000)
ax1.plot(line,line, color='grey')
ax1.set_title('Linear Model 2016-2017')
ax1.set_xlabel('True 2017 Home Values')
ax1.set_ylabel('Predicted 2017 Home Values')
ax1.set_xlim(150000, 700000)
ax1.set_ylim(150000, 700000)
plt.savefig('linear-models.png')
plt.show()

#------------------ RIDGE REGULARIZATION -------------------------------

#2a. Ridge 2016-2016
ridge = Ridge(alpha=1.0, normalize=True, max_iter=10000)
ridge.fit(X_train_std_2016, y_train_2016)

test_predicted_ridge_2016 = ridge.predict(X_test_std_2016)
rmse_test_ridge_2016 = rmse(y_test_2016, test_predicted_ridge_2016)

#2b. Ridge 2016-2017
predicted_ridge_2017 = ridge.predict(X_2017_std)
rmse_test_ridge_2017 = rmse(y_2017, predicted_ridge_2017)

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(28, 8))
ax0.scatter(y_test_2016, test_predicted_ridge_2016, alpha = .75, color='#229954')
line = np.linspace(150000, 700000, 1000)
ax0.plot(line,line, color='grey')
ax0.set_title('Ridge Model 2016-2016')
ax0.set_xlabel('True 2016 Home Values')
ax0.set_ylabel('Predicted 2016 Home Values')
ax0.set_xlim(150000, 700000)
ax0.set_ylim(150000, 700000)

ax1.scatter(y_2017, predicted_ridge_2017, alpha = .75, color='#229954')
line = np.linspace(150000, 700000, 1000)
ax1.plot(line,line, color='grey')
ax1.set_title('Ridge Model 2016-2017')
ax1.set_xlabel('True 2017 Home Values')
ax1.set_ylabel('Predicted 2017 Home Values')
ax1.set_xlim(150000, 700000)
ax1.set_ylim(150000, 700000)

plt.savefig('ridge-models.png')
plt.show()

#2c. Ridge alphas
ridge_alphas = np.logspace(-2, 4, num=250)

ridge_cv_errors_train, ridge_cv_errors_test = train_at_various_alphas(
    X_train_2016, y_train_2016, Ridge, ridge_alphas)
# print(ridge_cv_errors_train.head())
# print(ridge_cv_errors_test.head())

ridge_mean_cv_errors_train = ridge_cv_errors_train.mean(axis=0)
ridge_mean_cv_errors_test = ridge_cv_errors_test.mean(axis=0)

ridge_optimal_alpha = get_optimal_alpha(ridge_mean_cv_errors_test)

fig, ax = plt.subplots(figsize=(14, 4))

ax.plot(np.log10(ridge_alphas), ridge_mean_cv_errors_train, label="train")
ax.plot(np.log10(ridge_alphas), ridge_mean_cv_errors_test, label="test")
ax.axvline(np.log10(ridge_optimal_alpha), color='grey')
ax.set_title("Ridge Regression Train and Test RMSE")
ax.set_xlabel(r"$\log(\alpha)$")
ax.set_ylabel("RMSE")
plt.legend()
plt.savefig('ridge_alphas')
plt.show()

# #------------------ LASSO REGULARIZATION -------------------------------
#
# #3a. Lasso 2016-2016
lasso = Lasso(alpha=.8, normalize=True, max_iter=10000, selection='random')
lasso.fit(X_train_std_2016, y_train_2016)
test_predicted_lasso_2016 = lasso.predict(X_test_std_2016)
rmse_test_lasso_2016 = rmse(y_test_2016, test_predicted_lasso_2016)

#3b. Lasso 2016-2017
predicted_lasso_2017 = lasso.predict(X_2017_std)
rmse_test_lasso_2017 = rmse(y_2017, predicted_lasso_2017)

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(28, 8))
ax0.scatter(y_test_2016, test_predicted_lasso_2016, alpha = .75, color='#229954')
line = np.linspace(150000, 700000, 1000)
ax0.plot(line,line, color='grey')
ax0.set_title('Lasso Model 2016-2016')
ax0.set_xlabel('True 2016 Home Values')
ax0.set_ylabel('Predicted 2016 Home Values')
ax0.set_xlim(150000, 700000)
ax0.set_ylim(150000, 700000)

ax1.scatter(y_2017, predicted_lasso_2017, alpha = .75, color='#229954')
line = np.linspace(150000, 700000, 1000)
ax1.plot(line,line, color='grey')
ax1.set_title('Lasso Model 2016-2017')
ax1.set_xlabel('True 2017 Home Values')
ax1.set_ylabel('Predicted 2017 Home Values')
ax1.set_xlim(150000, 700000)
ax1.set_ylim(150000, 700000)

plt.savefig('lasso-models.png')
plt.show()

lasso_alphas = np.logspace(-3, 1, num=250)

lasso_cv_errors_train, lasso_cv_errors_test = train_at_various_alphas(
    X_train_2016, y_train_2016, Lasso, lasso_alphas, max_iter=5000)

lasso_mean_cv_errors_train  = lasso_cv_errors_train.mean(axis=0)
lasso_mean_cv_errors_test = lasso_cv_errors_test.mean(axis=0)

lasso_optimal_alpha = get_optimal_alpha(lasso_mean_cv_errors_test)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(np.log10(lasso_alphas), lasso_mean_cv_errors_train, label="train")
ax.plot(np.log10(lasso_alphas), lasso_mean_cv_errors_test, label="test")
ax.axvline(np.log10(lasso_optimal_alpha), color='grey')
ax.set_title("Lasso Regression Train and Test RMSE")
ax.set_xlabel(r"$\log(\alpha)$")
ax.set_ylabel("RMSE")
plt.legend()
plt.savefig('lasso_alphas')
plt.show()
