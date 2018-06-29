import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

data = pd.read_csv('~/galvanize/week4/analytic_capstone/main.csv')
df = data.copy()

#create dataframes for each year
col_2016 = [col for col in df if col.startswith('2016')]
col_2017 = [col for col in df if col.startswith('2017')]
col_2018 = [col for col in df if col.startswith('2018')]

df_2016 = df[col_2016]
df_2017 = df[col_2017]
df_2018 = df[col_2018]

#be able to explain why dropped these
df_2016.drop(['2016_Zip_Zhvi_AllHomes', '2016_Zip_Zhvi_BottomTier',
        '2016_Zip_Zhvi_MiddleTier', '2016_Zip_Zhvi_TopTier',
        '2016_Zip_MedianListingPricePerSqft_AllHomes',
        '2016_Zip_MedianValuePerSqft_AllHomes',
        '2016_Zip_PctOfListingsWithPriceReductions_AllHomes',
        '2016_Zip_Zri_AllHomesPlusMultifamily',
        '2016_Zip_ZriPerSqft_AllHomes'], inplace = True, axis = 1)
df_2017.drop(['2017_Zip_Zhvi_AllHomes', '2017_Zip_Zhvi_BottomTier',
        '2017_Zip_Zhvi_MiddleTier', '2017_Zip_Zhvi_TopTier',
        '2017_Zip_MedianListingPricePerSqft_AllHomes',
        '2017_Zip_MedianValuePerSqft_AllHomes',
        '2017_Zip_PctOfListingsWithPriceReductions_AllHomes',
        '2017_Zip_Zri_AllHomesPlusMultifamily',
        '2017_Zip_ZriPerSqft_AllHomes'], inplace = True, axis = 1)
df_2018.drop(['2018_Zip_Zhvi_AllHomes', '2018_Zip_Zhvi_BottomTier',
        '2018_Zip_Zhvi_MiddleTier', '2018_Zip_Zhvi_TopTier',
        '2018_Zip_MedianListingPricePerSqft_AllHomes',
        '2018_Zip_MedianValuePerSqft_AllHomes',
        '2018_Zip_PctOfListingsWithPriceReductions_AllHomes',
        '2018_Zip_Zri_AllHomesPlusMultifamily',
        '2018_Zip_ZriPerSqft_AllHomes'], inplace = True, axis = 1)
X_2016 = df_2016.values
X_2017 = df_2017.values
X_2018 = df_2018.values

y_2016 = df['2016_Zip_Zhvi_AllHomes'].values
y_2017 = df['2017_Zip_Zhvi_AllHomes'].values
y_2018 = df['2018_Zip_Zhvi_AllHomes'].values

## plotting y histograms
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(24,7))
ax0.hist(y_2016, 20, facecolor = '#5DADE2')
ax0.set_title('2016')
ax0.set_xlabel('Home Values')
ax0.set_xlim(200000, 700000)

ax1.hist(y_2017, 20, facecolor = '#FF5733')
ax1.set_title('2017')
ax1.set_xlabel('Home Values')
ax1.set_xlim(200000, 700000)

ax2.hist(y_2018, 20, facecolor = '#27AE60')
ax2.set_title('2018')
ax2.set_xlabel('Home Values')
ax2.set_xlim(200000, 700000)

plt.savefig('ZHVI2016-2018.png')
fig.tight_layout()
plt.show()


df_desc = pd.DataFrame({
                        'Home Values': [361752.69, 394906.94, 418970.56],
                        'Bottom Third Home Values': [260545.69, 291087.66, 311840.38],
                        'Top Third Home Values': [521847.50, 560900.20,  592481.10],
                        'Median List Price': [419982.44, 443638.92, 455585.07]},
                        index = ['2016', '2017', '2018'])
fig = plt.figure(figsize=(8, 5))
ax0 = fig.add_subplot(1,1,1)
ax0.plot(df_desc, marker="+", alpha=.75, lw=3)
ax0.legend(df_desc.columns, fontsize=12, loc='lower right')
ax0.set_ylim(120000, 600000)
fig.tight_layout()
plt.savefig('value_over_year.png')
plt.show()

df_inventory = df[['2016_InventoryMeasure_SSA_Zip_Public',
'2017_InventoryMeasure_SSA_Zip_Public',
'2018_InventoryMeasure_SSA_Zip_Public']]
# df_dec = df[['2016_Zip_MedianListingPrice_AllHomes',
#                 '2017_Zip_MedianListingPrice_AllHomes',
#                 '2018_Zip_MedianListingPrice_AllHomes']]

df_dec = df[['2016_Zip_MedianListingPrice_AllHomes',
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
# plt.savefig('images/zhvi_inventory.png')
# plt.show()
