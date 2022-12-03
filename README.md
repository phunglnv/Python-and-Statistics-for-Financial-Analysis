
# Using Python to Predict the market 





## Multiple linear regression model
Applying multiple linear regression to generate a signal for the growth of SPY (the exchange-traded fund tracking S&P 500). Choosing SPY because it's very suitable for trading frequently (cheap prices, low fees, volatility). 

This project will predict data price change of SPY when US market opens in the morning. The difference in timezone causes the Asian market information available for US market at its opening. 

Data from Aord, HSI, Nikkei, CAC40, DAX, S&P500, NASDAQ, DJI to predict the daily price change of SPY. 

![Capture 2](https://user-images.githubusercontent.com/105278875/204292519-9ef0fd90-fda4-4ab8-8907-70ea8efcee17.PNG)

Our responses variable is open price of SPY tomorrow minus today's open. With this response, we expect to make a prediction in the morning in US market. Based on those price changes, we decided whether to long or short. 

There are 8 predictors divided into 3 groups. 
- Group 1: One-day lag variables from US market: Open - Open last day (SPY, SP500, Nasdaq, Dji)
- Group 2: One-day lag variables from European market: Price at noon - Open last day (CAC40, Daxi)
- Group 3: One-day lag variables from Asian market: Close - Open (Aord, HSI, Nikkei)

## Data 
- All dataset has 6 columns. Open is the price at the beginning. 
- High, Low are the highest and lowest prices on that day. 
- Adj Close price is the adjusted closing price that has been amended to include any disputions and corporate actions that occured at anytime before the next day's open. 
- Volumne is the number of shares traded on that day. 
- This project only use open price.  

![Capture](https://user-images.githubusercontent.com/105278875/204292189-cc9b93d2-f95c-45ae-a6af-173d5f224384.PNG)

## (1) Data Muggling
- First, we generate an empty data frame and let the index be the same as the index of SPY.
- Then check whether do we have NaN values in indicepanel.

![Capture 3](https://user-images.githubusercontent.com/105278875/204293049-baaa0e90-ed98-4e9f-9f7a-539df728d5f7.PNG)

Missing values (NaN) appear due to 2 reasons: 
- When calculating the price change, NaN value is generated in the first row (one day lag) and the last row (one day in the future). 
- In different markets, they may have different holidays in which the markets are closed. It can be shown by computing numbers of NaN values in each column. 

Fom the data, Australia markets seems to have more holidays in one-year period. <br>

We need to handle NaN values first before viewing the model. 
## (2) Data Splitting
To make sure the model is consistent in future data, current data need to be split into 2 parts: 
- One is for building the model
- The other part is for testing the model to see if the model can still make reasonable prediction in this dataset 

![Capture 3](https://user-images.githubusercontent.com/105278875/204293295-7fe7c133-b8d2-4483-976e-4c92c39598e8.PNG)

Stock data is very noisy comparing to other static data like images. We will use the equal size of samples for both train and test. 

We assign 500 days at the test data and 500 days before the test at the training data. 

## (3) Explore the train data set
- Use the scatter matrix to get a pairwise scatterplot.
- From the output, we find that the predictors for Asian and European markets do have association with SPY, which have higher impacts than predictors of U.S. markets. 
- We need to compute correlation in order to see the association clearly. 

![Capture 5](https://user-images.githubusercontent.com/105278875/204293660-b4dd6425-f080-434c-8167-f5c1171ff198.PNG)
## (4) Check the correlation of each index between SPY

![Capture 6](https://user-images.githubusercontent.com/105278875/204293860-499fce0b-9b56-42b2-934e-25cbf007edf3.PNG)

- P-value for F-statistics = 2.16e-14 < 0.05, which indicates we rejects H0 and shows that our model includes useful predictors. 
- P-value of Aord = 0 means most of the predictors are not significant, except Aord. Another way to say is all other predictors are useless information of SPY. It may be because of multicollinarity.

## (5) Make Prediction
As the scatter chart shown, it does have positive correlation.

![Capture 7](https://user-images.githubusercontent.com/105278875/204294056-0e8ba202-62db-4723-a099-88be18754f5f.PNG)

## (6) Model  Evaluation
We can evaluate the model by comparing 2 statistics in train and test. <br>
We can measure the performance of our model using some statistical metrics 
- RMSE
- Adjusted R-square

If RMSE and Adjusted R-square in train is much better in train than in test dataset, the model is overfitting and we cannot implies this model to real market in the future. 

From the output, our model is not overfitted. 

![Capture 8](https://user-images.githubusercontent.com/105278875/204294441-db555702-d886-44e5-95c5-3d0b7d1340b7.PNG)
