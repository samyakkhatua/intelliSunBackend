# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from sklearn.model_selection import train_test_split

# # Load the dataset from the CSV file
# # data = pd.read_csv('gptdata10.csv')
# data = pd.read_csv('data_simulated_10k.csv')

# # Select the features and target
# X = data[['Temperature', 'Sunlight_Intensity', 'Humidity']]
# y = data[['Panel_Orientation', 'Panel_Position', 'Current_Generated']]

# # Train a random forest regressor model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Predict the optimal values of Panel_Orientation, Panel_Position, and Current_Generated for the input data point
# input_data = [[23, 400, 40]]
# predicted_values = model.predict(input_data)[0]

# # Extract the predicted values
# orientation, position, current = predicted_values

# # Calculate the metrics for the model
# y_pred = model.predict(X_test)
# r2 = r2_score(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# mae = mean_absolute_error(y_test, y_pred)

# # Print the results
# print(f"Predicted Panel Orientation: {orientation:.2f}")
# print(f"Predicted Panel Position: {position:.2f}")
# print(f"Predicted Current Generated: {current:.2f}")
# print(f"R2 Score: {r2:.2f}")
# print(f"Mean Squared Error: {mse:.2f}")
# print(f"Root Mean Squared Error: {rmse:.2f}")
# print(f"Mean Absolute Error: {mae:.2f}")

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV

# Load the dataset from the CSV file
data = pd.read_csv('output.csv')

# Select the features and target
X = data[['Temperature', 'Sunlight_Intensity', 'Humidity']]
y = data[['Panel_Orientation', 'Panel_Position', 'Current_Generated']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Train a random forest regressor model with hyperparameter tuning
model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Predict the optimal values of Panel_Orientation, Panel_Position, and Current_Generated for the input data point
input_data = [[17.973519724651315,927.5797867663118,24.14297054962471]]
# 224.33426449412502,65.41036621099023,202.10076337739156

# input_data = [[23, 400, 40]]
predicted_values = grid_search.predict(input_data)[0]

# Extract the predicted values
orientation, position, current = predicted_values

# Calculate the metrics for the model
y_pred = grid_search.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

# Print the results
print(f"Predicted Panel Orientation: {orientation:.2f}")
print(f"Predicted Panel Position: {position:.2f}")
print(f"Predicted Current Generated: {current:.2f}")
print(f"R2 Score: {r2:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Best Parameters: {grid_search.best_params_}")
