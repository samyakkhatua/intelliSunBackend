import pandas as pd
import numpy as np
from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import List

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# Define input data schema
class InputData(BaseModel):
    temperature: float
    sunlight_intensity: float
    humidity: float


# Load the dataset from the CSV file
data = pd.read_csv('data_simulated_10k.csv')

# Select the features and target
X = data[['Temperature', 'Sunlight_Intensity', 'Humidity']]
y = data[['Panel_Orientation', 'Panel_Position', 'Current_Generated']]

# Train a linear regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)


# function to calculate the predicted values for 'Panel_Orientation', 'Panel_Position', and 'Current_Generated'
def predict_current(data_point):
    # Use the trained model to predict the values for 'Panel_Orientation', 'Panel_Position', and 'Current_Generated'
    predicted_values = model.predict(data_point)[0]
    predicted_df = pd.DataFrame(predicted_values, index=[''], columns=['Panel_Orientation', 'Panel_Position', 'Current_Generated'])
    return predicted_df


# Define the FastAPI instance
app = FastAPI()

# origins = [
#     "http://127.0.0.1:5500",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Define the API endpoint
@app.post("/predict")
async def predict(input_data: InputData):
    data_point = pd.DataFrame({'Temperature': input_data.temperature, 'Sunlight_Intensity': input_data.sunlight_intensity,
                               'Humidity': input_data.humidity}, index=[0])

    predicted_values = predict_current(data_point)

    return predicted_values.to_dict('records')[0]
