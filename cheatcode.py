import csv
import pandas as pd
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


# Define input data schema
class InputData(BaseModel):
    temperature: float
    sunlight_intensity: float
    humidity: float


# Load the dataset from the CSV file
# data = pd.read_csv('gptdata10.csv')
data = pd.read_csv('data_simulated_10k.csv')

# Select the features and target
X = data[['Temperature', 'Sunlight_Intensity', 'Humidity']]
y = data[['Panel_Orientation', 'Panel_Position', 'Current_Generated']]

# Train a linear regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)


# Define the function to predict the output and calculate metrics
def predict(input_data: InputData):
    data_point = pd.DataFrame({'Temperature': input_data.temperature,
                               'Sunlight_Intensity': input_data.sunlight_intensity,
                               'Humidity': input_data.humidity}, index=[0])

    # Predict the optimal values of Panel_Orientation, Panel_Position, and Current_Generated
    predicted_values = model.predict(data_point)

    # Extract the predicted values
    orientation, position, current = predicted_values[0]

    # Calculate R-squared, MSE, RMSE, and MAE
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    # Return the predicted values and metrics
    # return {'Panel_Orientation': orientation, 'Panel_Position': position, 'Current_Generated': current,'R-squared': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    return orientation,position, current


# Test the function with sample input data
# input_data = InputData(temperature=23, sunlight_intensity=400, humidity=40)
# output_data = predict(input_data)
# print(output_data)


# Open the input file and create a CSV reader
with open('data_simulated_10k.csv', 'r') as input_file:
    reader = csv.reader(input_file)

    # Open the output file and create a CSV writer
    with open('output.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file)

        # Write the header row for the output file
        writer.writerow(['Temperature', 'Sunlight_Intensity', 'Humidity', 'Panel_Orientation', 'Panel_Position', 'Current_Generated'])

        # Loop through each row in the input file
        next(reader) # skip the header row
        for row in reader:
            # Get the input values from the current row
            Temperature, Sunlight_Intensity, Humidity = float(row[0]), float(row[1]), float(row[2])

            # Call the function to generate panel data
            input_data = InputData(temperature=Temperature, sunlight_intensity=Sunlight_Intensity, humidity=Humidity)
            # output_data = predict(input_data)
            Panel_Orientation, Panel_Position, Current_Generated = predict(input_data)

            # Write the input and output values to the output file
            writer.writerow([Temperature, Sunlight_Intensity, Humidity, Panel_Orientation, Panel_Position, Current_Generated])
            print([Temperature, Sunlight_Intensity, Humidity, Panel_Orientation, Panel_Position, Current_Generated])
