import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load the dataset
df = pd.read_csv('gptdata10.csv')

# Define the input and output variables
X = df.iloc[:, :-3].values
y = df.iloc[:, -3:].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train different models and make predictions
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

svm = SVR(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Evaluate the performance of each model
for i in range(y_test.shape[1]):
    print('Evaluation metrics for', df.columns[-3+i], ':')
    print('Linear Regression - R-squared:', r2_score(y_test[:,i], y_pred_lr[:,i]), 
          'MSE:', mean_squared_error(y_test[:,i], y_pred_lr[:,i]), 
          'MAE:', mean_absolute_error(y_test[:,i], y_pred_lr[:,i]))
    print('Decision Tree - R-squared:', r2_score(y_test[:,i], y_pred_dt[:,i]), 
          'MSE:', mean_squared_error(y_test[:,i], y_pred_dt[:,i]), 
          'MAE:', mean_absolute_error(y_test[:,i], y_pred_dt[:,i]))
    print('Random Forest - R-squared:', r2_score(y_test[:,i], y_pred_rf[:,i]), 
          'MSE:', mean_squared_error(y_test[:,i], y_pred_rf[:,i]), 
          'MAE:', mean_absolute_error(y_test[:,i], y_pred_rf[:,i]))
    print('Support Vector Machine - R-squared:', r2_score(y_test[:,i], y_pred_svm[:,i]), 
          'MSE:', mean_squared_error(y_test[:,i], y_pred_svm[:,i]), 
          'MAE:', mean_absolute_error(y_test[:,i], y_pred_svm[:,i]))
