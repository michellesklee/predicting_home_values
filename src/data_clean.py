import glob
import numpy as np
import pandas as pd
from functools import reduce
from fancyimpute import KNN

#Read all files in folder
files = glob.glob("data/*.csv")

df_list = []
for file in files:
    df_list.append(pd.read_csv(file))

def clean_df(lst):
    df_list2 = []
    #Drop all non-Denver cities
    for idx, df in enumerate(df_list):
        if 'Metro' in df.columns:
            df.drop(df[df.Metro != 'Denver'].index, inplace=True)
        elif 'City' in df.columns:
            df.drop(df[df.City != 'Denver'].index, inplace=True)
        else:
            pass
        df['City'] = 'Denver' #make sure all dataframes have a City column

        #Take average of 2016, 2017, 2018 data
        col_2016 = [col for col in df if col.startswith('2016')]
        df['2016'] = df[col_2016].mean(axis=1)
        col_2017 = [col for col in df if col.startswith('2017')]
        df['2017'] = df[col_2017].mean(axis=1)
        col_2018 = [col for col in df if col.startswith('2018')]
        df['2018'] = df[col_2018].mean(axis=1)

        #Dataframe with only columns of interest
        df2 = df[['City', 'RegionName', '2016', '2017', '2018']]

        #Clean column names
        features = files[idx].replace('.csv', '').replace('data/', '_')
        df2.columns = ['City', 'RegionName', '2016'+features, '2017'+features, '2018'+features]
        df_list2.append(df2)
    return df_list2

df_list2 = clean_df(df_list)
df_main = reduce(lambda left, right: pd.merge(left, right, on=['RegionName'],
                        how = 'outer'), df_list2)
df_main.drop(['City_x', 'City_y'], inplace=True, axis=1)

#Impute data using fancyimpute KNN
df_main.as_matrix()
df_filled = pd.DataFrame(KNN(3).complete(df_main))
df_filled.columns = df_main.columns

#Merge 2016 and 2017 business data
df_biz2016 = pd.read_csv('business_data/business_license2016.csv')
df_biz2017 = pd.read_csv('business_data/business_license2017.csv')
df_biz2016 = df_biz2016.merge(df_biz2017, on='RegionName', how='outer')
df_biz2016.fillna(0, inplace=True)

#Merge city and business data
df_filled = df_filled.merge(df_biz2016, on='RegionName')


df_filled.to_csv('~/galvanize/week4/analytic_capstone/main.csv')
