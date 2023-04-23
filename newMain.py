# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score, mean_squared_error

# # Load the dataset
# df = pd.read_csv('data_simulated_10k.csv')

# # Split the dataset into training and testing sets
# X = df[['Temperature', 'Sunlight_Intensity', 'Humidity']]
# y = df[['Panel_Orientation', 'Panel_Position']]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train and evaluate a Linear Regression model
# lr = LinearRegression()
# lr.fit(X_train, y_train)
# lr_y_pred = lr.predict(X_test)
# lr_r2_score = r2_score(y_test, lr_y_pred)
# lr_mse = mean_squared_error(y_test, lr_y_pred)

# # Train and evaluate a Decision Tree Regression model
# dtr = DecisionTreeRegressor()
# dtr.fit(X_train, y_train)
# dtr_y_pred = dtr.predict(X_test)
# dtr_r2_score = r2_score(y_test, dtr_y_pred)
# dtr_mse = mean_squared_error(y_test, dtr_y_pred)

# # Train and evaluate a Random Forest Regression model
# rfr = RandomForestRegressor()
# rfr.fit(X_train, y_train)
# rfr_y_pred = rfr.predict(X_test)
# rfr_r2_score = r2_score(y_test, rfr_y_pred)
# rfr_mse = mean_squared_error(y_test, rfr_y_pred)

# # Print the results
# print('Linear Regression R2 Score:', lr_r2_score)
# print('Linear Regression Mean Squared Error:', lr_mse)
# print('Decision Tree Regression R2 Score:', dtr_r2_score)
# print('Decision Tree Regression Mean Squared Error:', dtr_mse)
# print('Random Forest Regression R2 Score:', rfr_r2_score)
# print('Random Forest Regression Mean Squared Error:', rfr_mse)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Load the dataset
# df = pd.read_csv('synthetic_data.csv')
df = pd.read_csv('data_simulated_10k.csv')

# Convert 'Panel_Orientation' column from strings to integers
le = LabelEncoder()
df['Panel_Orientation'] = le.fit_transform(df['Panel_Orientation'])
df['Panel_Position'] = le.fit_transform(df['Panel_Position'])


# Split into input (X) and output (y) variables
X = df[['Temperature', 'Sunlight_Intensity', 'Humidity']]
y = df[['Panel_Orientation', 'Panel_Position']]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)

# Decision Tree Regression
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_r2 = r2_score(y_test, dt_pred)
dt_mse = mean_squared_error(y_test, dt_pred)

# Random Forest Regression
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)

# Print the results
print(f'Linear Regression R2 Score: {lr_r2}')
print(f'Linear Regression Mean Squared Error: {lr_mse}')
print(f'Decision Tree Regression R2 Score: {dt_r2}')
print(f'Decision Tree Regression Mean Squared Error: {dt_mse}')
print(f'Random Forest Regression R2 Score: {rf_r2}')
print(f'Random Forest Regression Mean Squared Error: {rf_mse}')


# Use the best model to make predictions
best_model = rf_model  # for example, we choose Random Forest
input_data = [[50, 800, 50]]  # example input data
predicted_output = best_model.predict(input_data)
print(f'Predicted Panel Orientation and Position: {predicted_output}')



