![top](http://leads.perfectstormnow.com/image/site/304/partner-588932c21a6f0.jpg)
# Predicting Denver Metro Home Values 2016 - 2017

## Objective
Predict home values based on real estate and business-related features in Denver Metro area.

## Background
Home appraisals and valuations are critical components of the home buying and selling process. These estimates may influence list and sale price, the amount of time a property is on the market, and the perceived health of an area. The real estate website [Zillow](https://www.zillow.com) assigns home value estimates for homes in the U.S., calculated with data regarding the home's location, recent sale prices of comparable homes, features of the home, and market conditions.

While the Zillow estimate has been criticized for [lacking accuracy](https://www.washingtonpost.com/news/where-we-live/wp/2014/06/10/how-accurate-is-zillows-zestimate-not-very-says-one-washington-area-agent/?noredirect=on&utm_term=.ac4b2039e5f1), it is still one of the most widely accessed and freely available home appraisal estimations. The goal of this project is to develop a model that predicts the Zillow Home Valuation Index (ZHVI) based on features collected by Zillow as well as business license data from the [City and County of Denver](https://www.denvergov.org/opendata/search?tag=business%20licenses). The number of new licenses issued by type (e.g., liquor, short term rentals) was examined due to its potential indication of [neighborhood]("https://www.stlouisfed.org/~/media/Files/PDFs/Community-Development/Research-Reports/NeighborhoodCharacteristics.pdf?la=en") and [economic]("https://www.citylab.com/life/2015/11/the-connection-between-vibrant-neighborhoods-and-economic-growth/417714/") growth.

## I. Exploratory Data Analysis
![2016-2018](https://github.com/michellesklee/analytic_capstone/blob/master/images/ZHVI2016-2018.png)

![scatter](https://github.com/michellesklee/analytic_capstone/blob/master/images/eda_scatter.png)

![eda](https://github.com/michellesklee/analytic_capstone/blob/master/images/value_over_year.png)

## II. Dataset Creation

### Feature Space
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

### Data Imputation
Missing real estate data were imputed using the k-nearest neighbors algorithm from [fancyimpute](https://github.com/iskandr/fancyimpute).
Missing business license data were assumed to indicate no licenses were issued, and thus were imputed with 0.
The two datasets were then merged based on zip code.

### III. Model Building
Data were separated into train and test subsets and explored on linear regression for 2016. This model was then used to predict home values on 2017.

![lin1](https://github.com/michellesklee/analytic_capstone/blob/master/images/linear-2016-2016.png)

![lin2](https://github.com/michellesklee/analytic_capstone/blob/master/images/linear-2016-2017.png)

A large difference between train RMSE (7520.87) and test RMSE (18987.79) suggested overfit of the model. Thus, the model was tested again with Ridge  and Lasso regularization.

### IV. Model Results

| Year        | Linear | Ridge           | Lasso  |
| ------------- |:---|:-------------:| -----:|
| 2016     | 18,987.79| 18,829.99 | 17,951.41 |
| 2017     | 90,195.72| 34,387.60      |   49,549.69 |


![ridge1](https://github.com/michellesklee/analytic_capstone/blob/master/images/ridge-2016-2016.png)

![ridge2](https://github.com/michellesklee/analytic_capstone/blob/master/images/ridge-2016-2017.png)

![ridge3](https://github.com/michellesklee/analytic_capstone/blob/master/images/ridge_alphas.png)

![lasso1](https://github.com/michellesklee/analytic_capstone/blob/master/images/lasso-2016-2016.png)

![lasso2](https://github.com/michellesklee/analytic_capstone/blob/master/images/lasso-2016-2017.png)

![lasso3](https://github.com/michellesklee/analytic_capstone/blob/master/images/lasso_alphas.png)

### V. Conclusion
Using real estate and business-related features, the linear model had some predictive ability predicting home values in 2016 and 2017. However, the number of features likely contributed to an overfit model. Regularization with Ridge and Lasso resulted in similar RMSE to the initial linear model for 2016, but better fit the model when predicting 2017 home values.


### VI. Next Steps
- Attempt to predict on available 2018 home values
- Fine tune the model, looking more closely at business license features in particular
- Look into GIS data


#### Data Sources:
[Zillow Research](https://www.zillow.com/research/data/)

[City and County of Denver Open Source Data](https://www.denvergov.org/opendata/)

#### References
[New businesses as an indicator of neighborhood health](https://www.stlouisfed.org/~/media/Files/PDFs/Community-Development/Research-Reports/NeighborhoodCharacteristics.pdf?la=en)

[New businesses as an indicator of neighborhood economic growth](https://www.citylab.com/life/2015/11/the-connection-between-vibrant-neighborhoods-and-economic-growth/417714/)

[Criticism of the Zestimate](https://www.washingtonpost.com/news/where-we-live/wp/2014/06/10/how-accurate-is-zillows-zestimate-not-very-says-one-washington-area-agent/?noredirect=on&utm_term=.ac4b2039e5f1)
