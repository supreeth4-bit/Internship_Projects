import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

#Download historical stock data
stock_symbol="AAPL"  #Can change this to any stock ticker you want
data=yf.download(stock_symbol, start="2020-01-01", end="2025-01-01")
#Keep only the closing price
#yfinance sometimes returns a multi-index column, so this keeps it simple
if isinstance(data.columns, pd.MultiIndex):
    close_prices=data[("Close", stock_symbol)]
else:
    close_prices=data["Close"]
#Create a simple feature: previous day's closing price
stock_df=pd.DataFrame({"Close": close_prices})
stock_df["Previous_Close"]=stock_df["Close"].shift(1)
#Remove missing row caused by shift
stock_df=stock_df.dropna()
#Define features (X) and target (y)
X=stock_df[["Previous_Close"]]
y=stock_df["Close"]
#Split data into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
#Train a simple Linear Regression model
model=LinearRegression()
model.fit(X_train, y_train)
#Make predictions
predictions=model.predict(X_test)
#Evaluate the model
mae=mean_absolute_error(y_test, predictions)
r2=r2_score(y_test, predictions)
print(f"Stock: {stock_symbol}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.4f}")
#Predict next day's price using the latest closing price
latest_close=stock_df["Close"].iloc[-1]
#next_day_prediction=model.predict([[latest_close]])[0]
next_input = pd.DataFrame({"Previous_Close": [latest_close]})
next_day_prediction = model.predict(next_input)[0]
print(f"Latest Close Price: {latest_close:.2f}")
print(f"Predicted Next Day Price: {next_day_prediction:.2f}")
#Plot actual vs predicted prices
plt.figure(figsize=(10,5))
plt.plot(y_test.index,y_test.values,label="Actual Price")
plt.plot(y_test.index,predictions,label="Predicted Price")
plt.title(f"{stock_symbol} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
