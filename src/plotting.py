import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

import matplotlib as mpl
mpl.rc('axes.spines', top = False, right = False)
mpl.rcParams.update({
    'font.size'           : 12.0,
    'axes.titlesize'      : 'medium',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'medium',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'large',

})

data = pd.read_csv('/Users/michellelee/galvanize/week4/analytic_capstone/data/main.csv')
df = data.copy()

#create dataframes for each year
col_2016 = [col for col in df if col.startswith('2016')]
col_2017 = [col for col in df if col.startswith('2017')]
col_2018 = [col for col in df if col.startswith('2018')]

df_2016 = df[col_2016]
df_2017 = df[col_2017]
df_2018 = df[col_2018]

y_2016 = df['2016_Zip_Zhvi_AllHomes'].values
y_2017 = df['2017_Zip_Zhvi_AllHomes'].values
y_2018 = df['2018_Zip_Zhvi_AllHomes'].values

## plotting y histograms
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(28, 8))
ax0.hist(y_2016, facecolor = '#5DADE2')
ax0.set_title('2016')
ax0.set_xlabel('Home Values')
ax0.set_xlim(200000, 700000)
ax0.set_ylim(0, 18)

ax1.hist(y_2017, facecolor = '#FF5733')
ax1.set_title('2017')
ax1.set_xlabel('Home Values')
ax1.set_xlim(200000, 700000)
ax1.set_ylim(0, 18)

ax2.hist(y_2018, facecolor = '#27AE60')
ax2.set_title('2018')
ax2.set_xlabel('Home Values')
ax2.set_xlim(200000, 700000)
ax2.set_ylim(0, 18)

# plt.savefig('ZHVI2016-2018.png')
# plt.show()


df_real_estate_features = df_2016[[
           '2016_InventoryMeasure_SSA_Zip_Public',
           '2016_Zip_Listings_PriceCut_SeasAdj_AllHomes',
           '2016_Zip_Median_PriceCut_Dollar_AllHomes',
           '2016_Zip_MedianListingPrice_AllHomes',
           '2016_Zip_MedianRentalPrice_AllHomes',
           '2016_Zip_PctOfHomesDecreasingInValues_AllHomes',
           '2016_Zip_PctOfHomesIncreasingInValues_AllHomes',
           '2016_Zip_PriceToRentRatio_AllHomes',
           '2016_Zip_Zri_AllHomes']]
df_real_estate_features.hist(figsize=(25,25))
plt.savefig('real_estate_hist')
plt.show

df_biz_features = df_2016[[
           '2016_Body_Art_Est_Permanent', '2016_Child_Care',
       '2016_Combined_License', '2016_Food_Retail', '2016_Food_Wholesale',
       '2016_Garage_Repair_of_Motor_Vehic', '2016_Kennel', '2016_Liquor',
       '2016_Medical_Marijuana', '2016_Parking_Lot,_Garage',
       '2016_Pedal_Cab_Company', '2016_Retail_Food_Establishment',
       '2016_Retail_Marijuana_', '2016_Second_Hand_Dealer',
       '2016_Short_Term_Rental', '2016_Swimming_Pool',
       '2016_Tree_Service_Company', '2016_Valet_Location_License',
       '2016_Waste_Hauler', '2016_Grand_Total']]
df_biz_features.hist(figsize=(25,25))
plt.savefig('biz_hist')
plt.show()

#scatter matrix
df_scatter = df_2016[(df_2016['2016_Grand_Total'] > 0)]
df_2016_scatter = df_scatter[['2016_Zip_Zhvi_AllHomes',
           '2016_InventoryMeasure_SSA_Zip_Public',
           '2016_Zip_Listings_PriceCut_SeasAdj_AllHomes',
           '2016_Zip_Median_PriceCut_Dollar_AllHomes',
           '2016_Zip_MedianListingPrice_AllHomes',
           '2016_Zip_MedianRentalPrice_AllHomes',
           '2016_Zip_PctOfHomesDecreasingInValues_AllHomes',
           '2016_Zip_PctOfHomesIncreasingInValues_AllHomes',
           '2016_Zip_PriceToRentRatio_AllHomes',
           '2016_Zip_Zri_AllHomes', '2016_Grand_Total']]
df_2016_scatter.columns = ['ZHVI', 'Inventory', '%Price_Cut', 'Med_Price_Cut', 'List_Price',
                           'Rent_Price', '%Dec_Value', '%Inc_Value', 'Rent_Ratio',
                           'ZRI', 'Business']

scatter_matrix(df_2016_scatter, alpha=0.75, figsize=(14, 14), diagonal='kde')
# plt.savefig('scatter_matrix_2016')
# plt.show()


#scatter matrix with business data
df_biz_scatter = df_scatter[['2016_Zip_Zhvi_AllHomes',
                             '2016_Body_Art_Est_Permanent',
                             '2016_Child_Care',
                             '2016_Combined_License',
                             '2016_Food_Retail',
                             '2016_Food_Wholesale',
                             '2016_Garage_Repair_of_Motor_Vehic',
                             '2016_Kennel',
                             '2016_Liquor',
                             '2016_Medical_Marijuana',
                             '2016_Parking_Lot,_Garage',
                             '2016_Pedal_Cab_Company',
                             '2016_Retail_Food_Establishment',
                             '2016_Retail_Marijuana_',
                             '2016_Second_Hand_Dealer',
                             '2016_Short_Term_Rental',
                             '2016_Swimming_Pool',
                             '2016_Tree_Service_Company',
                             '2016_Valet_Location_License',
                             '2016_Waste_Hauler']]
df_biz_scatter.columns = ['ZHVI', 'Body_Art', 'Child_Care', 'Combined',
                          'Food_Retail', 'Food_Wholesale', 'Garage', 'Kennel', 'Liquor',
                          'Med_Marijuana', 'Parking_Lot', 'Pedal_Cab', 'Food_Estab', 'Retail_Marijuana',
                          'Second_Hand', 'Short_Term_Rental', 'Swimming_Pool', 'Tree_Service', 'Valet',
                          'Waste_Hauler']
scatter_matrix(df_biz_scatter, alpha=0.75, figsize=(14, 14), diagonal='kde')
plt.savefig('scatter_biz')
plt.show()

#some select scatter graphs
df_inventory = df[['2016_InventoryMeasure_SSA_Zip_Public',
                   '2017_InventoryMeasure_SSA_Zip_Public',
                   '2018_InventoryMeasure_SSA_Zip_Public']]

df_list = df[['2016_Zip_MedianListingPrice_AllHomes',
              '2017_Zip_MedianListingPrice_AllHomes',
              '2018_Zip_MedianListingPrice_AllHomes']]

df_rental = df[['2016_Zip_Zri_AllHomes', '2017_Zip_Zri_AllHomes', '2018_Zip_Zri_AllHomes']]

df_zhvi = df[['2016_Zip_Zhvi_AllHomes', '2017_Zip_Zhvi_AllHomes', '2018_Zip_Zhvi_AllHomes']]

fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(26,10))

ax0.scatter(df_zhvi, df_rental, alpha=.75, c="#76448A")
ax0.set_xlabel('Home Values')
ax0.set_ylabel('Rental Values')

ax1.scatter(df_zhvi, df_inventory, alpha=.75, c="#0E6251")
ax1.set_xlabel('Home Values')
ax1.set_ylabel('Inventory')

ax2.scatter(df_zhvi, df_dec, alpha=.75, c="#5D6D7E")
ax2.set_xlabel('Home Values')
ax2.set_ylabel('List Price')
plt.show()
plt.savefig('zhvi_inventory.png')
plt.show()
