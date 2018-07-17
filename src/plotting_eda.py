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
from sklearn.base import clone
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

col_names = ['Inventory', 'Listings_PriceCut', 'Median_PriceCut', 'MedianListPrice','MedianRentalPrice', '%DecreasingValue', '%IncreasingValue', 'PricetoRentRatio', 'ZHVI', 'ZRI_All', 'ZRI_Multifamily', 'ZRI_SingleFamily', 'Body_Art', 'Child_Care', 'Combined_License', 'Food_Retail', 'Food_Wholesale', 'Garage', 'Kennel', 'Liquor', 'Med_Marijuana', 'Parking_Lot', 'Pedal_Cab', 'Retail_Food_Estab', 'Retail_Marijuana', 'Second_Hand', 'Short_Term_Rental', 'Swimming_Pool', 'Tree_Service', 'Valet', 'Waste_Hauler', 'Total']

df_2016.columns = col_names
df_2017.columns = col_names
df_2018.columns = col_names[0:12]

df_2016.drop('ZHVI', inplace = True, axis = 1)
df_2017.drop('ZHVI', inplace = True, axis = 1)
df_2018.drop('ZHVI', inplace = True, axis = 1)

X_2016 = df_2016.values
X_2017 = df_2017.values
X_2018 = df_2018.values

y_2016 = df['2016_Zip_Zhvi_AllHomes'].values
y_2017 = df['2017_Zip_Zhvi_AllHomes'].values
y_2018 = df['2018_Zip_Zhvi_AllHomes'].values

#------------------ EDA: DATA VIZUALIZATION -------------------------------

# HISTOGRAMS OF HOME VALUES BY YEAR
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(28, 8))
ax0.hist(y_2016, facecolor = '#7DCEA0')
ax0.set_title('2016')
ax0.set_xlabel('Home Values')
ax0.set_xlim(200000, 700000)
ax0.set_ylim(0, 18)

ax1.hist(y_2017, facecolor = '#229954')
ax1.set_title('2017')
ax1.set_xlabel('Home Values')
ax1.set_xlim(200000, 700000)
ax1.set_ylim(0, 18)

ax2.hist(y_2018, facecolor = '#145A32')
ax2.set_title('2018')
ax2.set_xlabel('Home Values')
ax2.set_xlim(200000, 700000)
ax2.set_ylim(0, 18)
plt.savefig('images/ZHVI2016-2018.png')
plt.show()

# HISTOGRAM OF REAL ESTATE FEATURES
df_real_estate_features = df_2016[['Inventory', 'Listings_PriceCut', 'Median_PriceCut',
                                   'MedianListPrice', 'MedianRentalPrice', '%DecreasingValue',
                                   '%IncreasingValue', 'PricetoRentRatio', 'ZRI_All',
                                   'ZRI_Multifamily', 'ZRI_SingleFamily']]
df_real_estate_features.hist(figsize=(25,25), color='#2980B9')
plt.savefig('images/real_estate_hist')
plt.show()

# HISTOGRAM OF BUSINESS LICENSES
df_biz_features = df_2016[['Body_Art', 'Child_Care', 'Combined_License', 'Food_Retail',
       'Food_Wholesale', 'Garage', 'Kennel', 'Liquor', 'Med_Marijuana',
       'Parking_Lot', 'Pedal_Cab', 'Retail_Food_Estab', 'Retail_Marijuana',
       'Second_Hand', 'Short_Term_Rental', 'Swimming_Pool', 'Tree_Service',
       'Valet', 'Waste_Hauler', 'Total']]
df_biz_features.hist(figsize=(25,25), color='#884EA0')
plt.savefig('images/biz_hist')
plt.show()
