![top](http://leads.perfectstormnow.com/image/site/304/partner-588932c21a6f0.jpg)
# Predicting Denver Metro Home Values 2016 - 2017

## Objective
Predict home values based on real estate and business-related features in Denver Metro area.

## Background
Home appraisals and valuations are critical components of the home buying and selling process. These estimates may influence list and sale price, the amount of time a property is on the market, and the perceived health of an area. The real estate website [Zillow](https://www.zillow.com) assigns home value estimates for homes in the U.S., calculated with data regarding the home's location, recent sale prices of comparable homes, features of the home, and market conditions.

While the Zillow estimate has been criticized for [lacking accuracy](https://www.washingtonpost.com/news/where-we-live/wp/2014/06/10/how-accurate-is-zillows-zestimate-not-very-says-one-washington-area-agent/?noredirect=on&utm_term=.ac4b2039e5f1), it is still one of the most widely accessed and freely available home appraisal estimations. The goal of this project is to develop a model that predicts the Zillow Home Valuation Index (ZHVI) based on features collected by Zillow as well as business license data from the [City and County of Denver](https://www.denvergov.org/opendata/search?tag=business%20licenses). The number of new licenses issued by type (e.g., liquor, short term rentals) was examined due to its potential indication of [neighborhood]("https://www.stlouisfed.org/~/media/Files/PDFs/Community-Development/Research-Reports/NeighborhoodCharacteristics.pdf?la=en") and [economic]("https://www.citylab.com/life/2015/11/the-connection-between-vibrant-neighborhoods-and-economic-growth/417714/") growth.

## I. Feature Selection
From approximately 50 datasets with real estate data from Zillow, 9 features were selected after removing redundant features and initial tests of collinearity. From approximately 30 types of business licenses issued by the city of Denver, 17 were selected after removing features with fewer than 5 licenses for both 2016 and 2017.

##### Real Estate
1. Median list price
2. Listings with price cut
3. Median price cut
4. Median rental price
5. % homes decreasing in value
6. % homes increasing in value
7. Price to rent ratio
8. Rental values: multi-family
9. Rental values: single-family

##### Business Licenses
1. Body art
2. Child care
3. Combined use
4. Food retail
5. Food wholesale
6. Garage/motor repair
7. Kennel
8. Liquor
9. Medical marijuana
10. Parking lot
11. Pedal cab
12. Retail food establishment
13. Retail marijuana
14. Second hand dealer
15. Short term rental
16. Tree service company
17. Waste hauler

## II. Exploratory Data Analysis

**Home Values 2016-2018**

![2016-2018](https://github.com/michellesklee/analytic_capstone/blob/master/images/ZHVI2016-2018.png)

**Distribution of Real Estate Features**

![real_estate](https://github.com/michellesklee/analytic_capstone/blob/master/images/real_estate_hist.png)

**Distribution of Business License Features**

![business](https://github.com/michellesklee/analytic_capstone/blob/master/images/biz_hist.png)


### Data Imputation
Missing real estate data were imputed using the k-nearest neighbors algorithm from [fancyimpute](https://github.com/iskandr/fancyimpute).
Missing business license data were assumed to indicate no licenses were issued, and thus were imputed with 0.
The two datasets were then merged based on zip code.

### III. Model Building
Data were separated into train and test subsets and explored on linear regression for 2016. This model was then used to predict home values on 2017.

#### Linear Regression
![lin](https://github.com/michellesklee/analytic_capstone/blob/master/images/linear-models.png)

**Coefficients of linear model**

| Real Estate Features        | Coefficient |
 ------------- |--- |
| Rental_Value_Index_AllHomes    | 69,690.73
| Price_to_Rent_Ratio    | 30,077.58
| %Increasing_Value     | 19,012.93
| %Decreasing_Value     | 17,449.90
| Rent_Price     | 7,662.12
| List_Price     | 4,321.72
| Rental_Value_Index_Single_Family    | 2,616.27
| Median_Price_Cut    | 2,404.02
| %Price_Cut     | -5,193.54
| %Rental_Value_Index_Multifamily    | -3,622.14
| Inventory     | -1,865.00


| Business Licenses        | Coefficient |
 ------------- |--- |
| Child_Care     | 12,437.47
| Food_Retail    | 5,199.60
| Garage_Repair_of_Motor_Vehicles    | 4,470.63
| Parking_Lot    | 4,033.58
| Food_Wholesale     | 3,838.60
| Second_Hand_Dealer  | 3,272.74
| Tree_Service_Company |2,783.19
| Retail_Marijuana  | 2,741.50
|Total|2,033.61
| Waste_Hauler |1,804.13
| Combined_License    | 1,032.17
| Swimming_Pool| -642.33
| Pedal_Cab_Company   | -1,155.10
| Liquor    | -1,821.86
| Valet |-2,243.60
| Kennel     | -5,242.89
| Medical_Marijuana    | -6,906.41
| Body_Art     | -11,804.17
| Retail_Food_Establishment   | -12,104.09


A large difference between train RMSE (7520.87) and test RMSE (18987.79) suggested overfit of the model. Thus, the model was tested again with Ridge  and Lasso regularization.

### IV. Model Results

![ridge](https://github.com/michellesklee/analytic_capstone/blob/master/images/ridge-models.png)

![ridge_mse](https://github.com/michellesklee/analytic_capstone/blob/master/images/ridge_alphas.png)

![lasso](https://github.com/michellesklee/analytic_capstone/blob/master/images/lasso-models.png)

![lasso_mse](https://github.com/michellesklee/analytic_capstone/blob/master/images/lasso_alphas.png)

**RMSE across all models**

| Year        | Linear | Ridge           | Lasso  |
| ------------- |:---|:-------------:| -----:|
| 2016     | 18,987.79| 18,829.99 | 17,951.41 |
| 2017     | 90,195.72| 34,387.60      |   49,549.69 |


### V. Additional Models
#### Feature Importance
Due to the number and collinearity of features, RFE on sklearn was conducted to select the most important features and the models were evaluated again.


| Feature Rank      | Feature |
 ------------- |--- |
| 1    | Food_Wholesale
| 1    | Garage
| 1    | Retail_Food_Estab
| 1    | Second_Hand
| 1    | Tree_Service_Company
| 1    | Listings_PriceCut
| 1    | Retail_Marijuana
| 1    | %DecreasingValue
| 1    | %IncreasingValue
| 1    | PricetoRentRatio
| 1    | ZRI_All
| 1    | ZRI_SingleFamily
| 2    | Listings_PriceCut
| 3    | MedianRentalPrice
| 4    | ZRI_Multifamily
| 5    | Short_Term_Rental

**Results**

| Year        | Linear | Ridge           | Lasso  |
| ------------- |:---|:-------------:| -----:|
| 2016     | 17,577.96| 30,543.88 | 16,350.88 |
| 2017     | 15,034.55| 21,392.57      |   19,990.87 |


The linear and lasso models with feature selection perform comparably as the original model but are much better at predicting 2017 home values.

#### Real Estate Features
To see if real estate features, which are highly correlated with home values, are the primary "drivers" are the models, RMSEs were calculated again with only real estate features.

| Year        | Linear | Ridge           | Lasso  |
| ------------- |:---|:-------------:| -----:|
| 2016     | 14828.44| 33274.60 |14966.75 |
| 2017     | 14717.81| 38787.34     |  12393.56 |

**Results**

The models with only real estate features in fact perform very similarly with models that include business license features!


### VI. Conclusion
Using real estate and business-related features, the linear model had some predictive ability predicting home values in 2016 and 2017. However, the number of features likely contributed to an overfit model. Regularization with Ridge and Lasso resulted in similar RMSE to the initial linear model for 2016, but better fit the model when predicting 2017 home values.

When comparing the original models with feature selection and real estate features only, the original models performed similarly, suggesting that business license features warrant further investigation.


### VII. Next Steps
- Attempt to predict on available 2018 home values
- Fine tune the model
- Look into GIS data


#### Data Sources:
[Zillow Research](https://www.zillow.com/research/data/)

[City and County of Denver Open Source Data](https://www.denvergov.org/opendata/)

#### References
[New businesses as an indicator of neighborhood health](https://www.stlouisfed.org/~/media/Files/PDFs/Community-Development/Research-Reports/NeighborhoodCharacteristics.pdf?la=en)

[New businesses as an indicator of neighborhood economic growth](https://www.citylab.com/life/2015/11/the-connection-between-vibrant-neighborhoods-and-economic-growth/417714/)

[Criticism of the Zestimate](https://www.washingtonpost.com/news/where-we-live/wp/2014/06/10/how-accurate-is-zillows-zestimate-not-very-says-one-washington-area-agent/?noredirect=on&utm_term=.ac4b2039e5f1)
