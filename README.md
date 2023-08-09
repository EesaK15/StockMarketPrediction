# StockMarketPrediction
"Stock Price Prediction using Yahoo Finance: Leveraging the power of Yahoo Finance data, this project delves into the art of predicting stock prices through historical data analysis. Discover how machine learning techniques can uncover patterns and trends in stock price movements, enabling accurate predictions for informed decision-making. 
### Installing Yahoo Finance
```
pip install yfinance 
```
Install the Yahoo Finance Library if not already done 
### Importing Libraries
```
import pandas as pd
import yfinance as yf
```
### Importing the SP500 statistics 
```
sp500 = yf.Ticker('^GSPC')
sp500 = sp500.history(period = 'max') # this will query through all the historical data from the beginning
print(sp500)

# each row is the price on a trading day
# columns are opening price, the highest it hit, the lowest and the closing, and total volume 
# using these columns, we'll predict the price the next day
```
sp500 = yf.Ticker('^GSPC'): This line initializes a Ticker object from Yahoo Finance using the symbol ^GSPC, which is the ticker symbol for the S&P 500 index
sp500 = sp500.history(period='max'): This line fetches the historical data for the S&P 500 index using the history method of the Ticker object; this will return the maximum available historical data

```
sp500.index
```
Provides a list of datetime values corresponding to the dates for which historical data is available in the DataFrame.


### Cleaning and Visualizing Our Stock Market Data
```
sp500.plot.line(y='Close', use_index = True)

# We may also remove unncessary columns:
del sp500['Dividends']
del sp500['Stock Splits']

# setting up our target for machine learning

# can we predict the price will go up
sp500['Tomorrow'] = sp500['Close'].shift(-1)
# Created a Tomorrow column, and have the value of the closing price in the tomorrow column for one row up (shift -1)

print(sp500)
sp500['Target'] = (sp500['Tomorrow']>sp500['Close']).astype(int)

print(sp500)

# 1 if True, so if price went up
# 0 if False, so if price went down

```
If the tomorrow value is greater than the value of the closing of the previous day, we may assign the number to indicate a growth in price. This would mean that it is a great day to sell the stock purchased the day before. 
```
# For greater accuracy in current events, well remove all data before 1990

sp500 = sp500.loc['1990-01-01':].copy()
print(sp500)
```

### Data Training
```
from sklearn.ensemble import RandomForestClassifier
# RandomForestClassifier - trains individual trees with random results and averages it 
# can pick up non linear tendancies

model = RandomForestClassifier(n_estimators = 100, min_samples_split = 100, random_state = 1)

train = sp500.iloc[:-100] # start until final 100 rows
test = sp500.iloc[-100:] # start from final 100 to end

# split the data, and since it is time series, we'll have to split it off a certain data

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])
```
This parameter defines the minimum number of samples required to split an internal node of each decision tree. It helps control the complexity of individual trees and can prevent overfitting. In your case, min_samples_split = 100 indicates that a node will only be split if it has at least 100 samples.

### Checking Accuracy
```
from sklearn.metrics import precision_score
preds = model.predict(test[predictors])

preds = pd.Series(preds, index = test.index)
precision_score(test['Target'],preds) # precision of 58
combined = pd.concat([test['Target'], preds], axis = 1)
combined.plot()
```

### Back Testing
```
# Building a back test system

def predict(train, test, predictors, model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name = 'Predictions')
    combined = pd.concat([test['Target'], preds], axis= 1)
    return combined
def backtest(data, model, predictors, start = 2500, step = 250):
    #training 10 years of data
    all_predictions = []
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train,test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)
```

A backtesting system is a method used to evaluate the performance of a trading or investment strategy using historical data. It's a crucial component of quantitative finance and algorithmic trading, allowing traders and investors to assess how well a strategy would have performed in the past before applying it to real-world trading.

```
predictions = backtest(sp500, model, predictors)
predictions['Predictions'].value_counts()
precision_score (predictions['Target'], predictions['Predictions'])
# We predicted the market would go down 3433 days and go up 2519
```


