![top](http://leads.perfectstormnow.com/image/site/304/partner-588932c21a6f0.jpg)
# Predicting Denver Metro Home Values 2016 - 2017

## Objective
Predict home values based on real estate and business-related features in Denver Metro area.

## Background
Home appraisals and valuations are critical factors in the home buying and selling process. These estimates may influence the list and sale price, the amount of time a property is on the market, and the perceived health of an area. The real estate website [Zillow](https://www.zillow.com) assigns home value estimates for homes in the U.S., calculated with data regarding the home's location, features, and market conditions.

While the Zillow estimate has been criticized for [lacking accuracy](https://www.washingtonpost.com/news/where-we-live/wp/2014/06/10/how-accurate-is-zillows-zestimate-not-very-says-one-washington-area-agent/?noredirect=on&utm_term=.ac4b2039e5f1), it is still one of the most widely accessed and freely available home appraisal estimations. The goal of this current project was to develop a model that predicts the Zillow Home Valuation Index (ZHVI) based on features collected by Zillow as well as business license data from the [City and County of Denver](https://www.denvergov.org/opendata/search?tag=business%20licenses).

## Exploratory Data Analysis
### Part I. Dataset Creation

#### Feature Space
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

#### Data Imputation
Missing real estate data were imputed using the k-nearest neighbors algorithm from [fancyimpute](https://github.com/iskandr/fancyimpute).
Missing business license data were assumed to indicate no licenses were issued, and thus were imputed with 0. 
The two datasets were then merged based on zip code.

### Part II. Data Visualization
![2017]("https://github.com/michellesklee/analytic_capstone/blob/master/images/ZHVI-2016.png")
![2017](https://github.com/michellesklee/analytic_capstone/blob/master/images/ZHVI-2017.png)

![value_over_year](https://github.com/michellesklee/analytic_capstone/blob/master/images/value_over_year.png)


