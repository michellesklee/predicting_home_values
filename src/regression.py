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

style.use('ggplot')
import matplotlib
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)


def standardize_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std, y_train, y_test

def rmse(true, predicted):
    return np.sqrt(np.mean((true - predicted) ** 2))

def cv(X, y, base_estimator, n_folds, random_seed=150):
    kf = KFold(n_splits=n_folds, random_state=random_seed)
    test_cv_errors, train_cv_errors = np.empty(n_folds), np.empty(n_folds)
    for idx, (train, test) in enumerate(kf.split(X_train)):
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

def linear_model(X, y):
    X_train_std, X_test_std, y_train, y_test = standardize_data(X, y)
    linear = LinearRegression()
    linear.fit(X_train_std, y_train)
    train_predicted = linear.predict(X_train_std)
    test_predicted = linear.predict(X_test_std)
    rmse_train = rmse(y_train, train_predicted)
    rmse_test = rmse(y_test, test_predicted)
    print("RMSE of train set for linear model is {}".format(rmse_train))
    print("RMSE of test set for linear model is {}".format(rmse_test))
    #return rmse_train, rmse_test

def linear_coefs(X, y, df):
    X_train_std, X_test_std, y_train, y_test = standardize_data(X, y)
    linear = LinearRegression()
    linear.fit(X_train_std, y_train)
    #Coefficients of linear model
    linear_coefs = list(linear.coef_)
    lin_col_names = list(df)
    col_dict = dict(zip(df, linear_coefs))
    return col_dict

def ridge_model(X, y):
    X_train_std, X_test_std, y_train, y_test = standardize_data(X, y)
    ridge = Ridge(alpha=1.0, normalize=True, max_iter=10000)
    ridge.fit(X_train_std, y_train)
    test_predicted_ridge = ridge.predict(X_test_std)
    rmse_test_ridge = rmse(y_test, test_predicted_ridge)
    print("RMSE of test set for ridge model is {}".format(rmse_test_ridge))
    #return rmse_test_ridge

def lasso_model(X, y):
    X_train_std, X_test_std, y_train, y_test = standardize_data(X, y)
    lasso = Lasso(alpha=.8, normalize=True, max_iter=10000, selection='random')
    lasso.fit(X_train_std, y_train)
    test_predicted_lasso = lasso.predict(X_test_std)
    rmse_test_lasso = rmse(y_test, test_predicted_lasso)
    #return rmse_test_lasso
    print("RMSE of test set for lasso model is {}".format(rmse_test_lasso))

def rfe(X, y, model, df):
    X_train_std, X_test_std, y_train, y_test = standardize_data(X, y)
    selector = RFE(model)
    selector.fit(X_train_std, y_train)
    column_ranking = []
    lin_col_names = list(df)
    ranking = list(selector.ranking_)
    for rank, col in zip(ranking, lin_col_names):
        column_ranking.append((rank, col))
    column_ranking.sort()
    return column_ranking

if __name__ == '__main__':

    data = pd.read_csv('/Users/michellelee/galvanize/week4/analytic_capstone/data/main.csv')
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

    linear = LinearRegression()
    #RMSEs for 2016-2016 and 2016-2017 data
    # print(linear_model(X_2016, y_2016))
    # print(linear_model(X_2016, y_2017))
    # print(ridge_model(X_2016, y_2016))
    # print(ridge_model(X_2016, y_2017))
    # print(lasso_model(X_2016, y_2016))
    # print(lasso_model(X_2016, y_2017))

    # #coefficients of linear model
    # print("Coefficients of linear model: {}".format(linear_coefs(X_2016, y_2016, df_2016)))

    #RFE
    #print(rfe(X_2016, y_2016, linear, df_2016))
    df_rfe_2016 = df_2016[['2016_Combined_License',
     '2016_Food_Wholesale',
     '2016_Garage_Repair_of_Motor_Vehic',
     '2016_Retail_Food_Establishment',
     '2016_Second_Hand_Dealer',
     '2016_Tree_Service_Company',
     '2016_Zip_Listings_PriceCut_SeasAdj_AllHomes',
     '2016_Zip_PctOfHomesDecreasingInValues_AllHomes',
     '2016_Zip_PctOfHomesIncreasingInValues_AllHomes',
     '2016_Zip_PriceToRentRatio_AllHomes',
     '2016_Zip_Zri_AllHomes',
     '2016_Zip_Zri_SingleFamilyResidenceRental',
     '2016_Zip_MedianRentalPrice_AllHomes',
     '2016_Zip_Zri_MultiFamilyResidenceRental',
     '2016_Short_Term_Rental']]
    df_rfe_2017 = df_2017[['2017_Combined_License',
     '2017_Food_Wholesale',
     '2017_Garage_Repair_of_Motor_Vehic',
     '2017_Retail_Food_Establishment',
     '2017_Second_Hand_Dealer',
     '2017_Tree_Service_Company',
     '2017_Zip_Listings_PriceCut_SeasAdj_AllHomes',
     '2017_Zip_PctOfHomesDecreasingInValues_AllHomes',
     '2017_Zip_PctOfHomesIncreasingInValues_AllHomes',
     '2017_Zip_PriceToRentRatio_AllHomes',
     '2017_Zip_Zri_AllHomes',
     '2017_Zip_Zri_SingleFamilyResidenceRental',
     '2017_Zip_MedianRentalPrice_AllHomes',
     '2017_Zip_Zri_MultiFamilyResidenceRental',
     '2017_Short_Term_Rental']]
    X_2016_rfe = df_rfe_2016.values
    X_2017_rfe = df_rfe_2017.values
    # print(linear_model(X_2016_rfe, y_2016))
    # print(linear_model(X_2016_rfe, y_2017))
    # print(ridge_model(X_2016_rfe, y_2016))
    # print(ridge_model(X_2016_rfe, y_2017))
    # print(lasso_model(X_2016_rfe, y_2016))
    # print(lasso_model(X_2016_rfe, y_2017))

    #Just real estate features
    df_re_2016 = df_2016[['2016_InventoryMeasure_SSA_Zip_Public',
       '2016_Zip_Listings_PriceCut_SeasAdj_AllHomes',
       '2016_Zip_Median_PriceCut_Dollar_AllHomes',
       '2016_Zip_MedianListingPrice_AllHomes',
       '2016_Zip_MedianRentalPrice_AllHomes',
       '2016_Zip_PctOfHomesDecreasingInValues_AllHomes',
       '2016_Zip_PctOfHomesIncreasingInValues_AllHomes',
       '2016_Zip_PriceToRentRatio_AllHomes', '2016_Zip_Zri_AllHomes',
       '2016_Zip_Zri_MultiFamilyResidenceRental',
       '2016_Zip_Zri_SingleFamilyResidenceRental']]
    df_re_2017 = df_2017[['2017_InventoryMeasure_SSA_Zip_Public',
       '2017_Zip_Listings_PriceCut_SeasAdj_AllHomes',
       '2017_Zip_Median_PriceCut_Dollar_AllHomes',
       '2017_Zip_MedianListingPrice_AllHomes',
       '2017_Zip_MedianRentalPrice_AllHomes',
       '2017_Zip_PctOfHomesDecreasingInValues_AllHomes',
       '2017_Zip_PctOfHomesIncreasingInValues_AllHomes',
       '2017_Zip_PriceToRentRatio_AllHomes', '2017_Zip_Zri_AllHomes',
       '2017_Zip_Zri_MultiFamilyResidenceRental',
       '2017_Zip_Zri_SingleFamilyResidenceRental']]
    X_2016_re = df_re_2016.values
    X_2017_re = df_re_2017.values
    print(linear_model(X_2016_re, y_2016))
    print(linear_model(X_2016_re, y_2017))
    print(ridge_model(X_2016_re, y_2016))
    print(ridge_model(X_2016_re, y_2017))
    print(lasso_model(X_2016_re, y_2016))
    print(lasso_model(X_2016_re, y_2017))
