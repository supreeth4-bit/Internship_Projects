import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

#Load weather dataset
#For other datasets,replace 'weather_data.csv' with your file name
#CSV should have columns:Date,Temperature

data=pd.read_csv("weather_data.csv")
#Convert Date column to datetime
data["Date"] = pd.to_datetime(data["Date"])
#Sort data by date
data=data.sort_values("Date")
#Create a numeric feature from date (number of days since start)
data["Days"]=(data["Date"] - data["Date"].min()).dt.days
#Define features and target
X=data[["Days"]]
y=data["Temperature"]
#Split data into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
#Train Linear Regression model
model=LinearRegression()
model.fit(X_train,y_train)
#Make predictions on test data
predictions=model.predict(X_test)
#Evaluate model
mae=mean_absolute_error(y_test, predictions)
r2=r2_score(y_test, predictions)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.4f}")
#Predict future temperature (next 7 days)
last_day=data["Days"].max()
future_days=pd.DataFrame({"Days":np.arange(last_day+1,last_day+8)})
future_predictions=model.predict(future_days)
print("\nPredicted Temperatures for Next 7 Days:")
for i, temp in enumerate(future_predictions, start=1):
    print(f"Day {i}:{temp:.2f}°C")

#Plot actual data and regression line
plt.figure(figsize=(10, 5))
plt.scatter(data["Date"], data["Temperature"], label="Actual Temperature")
plt.plot(data["Date"],model.predict(X),label="Trend Line")
plt.title("Weather Temperature Trend Prediction")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()
