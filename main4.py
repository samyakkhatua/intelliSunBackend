import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
X = data[['Temperature', 'Sunlight_Intensity', 'Humidity', 'Panel_Orientation', 'Panel_Position']]
y = data['Current_Generated']

# Train a linear regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)


# Define a function to calculate the predicted current generated by the solar panel
def predict_current(data_point):
    # Use the trained model to predict the current generated by the solar panel
    current = model.predict(data_point)[0]
    return current


# Define the FastAPI instance
app = FastAPI()

origins = [
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the API endpoint
@app.post("/predict")
async def predict(input_data: InputData):
    data_point = pd.DataFrame({'Temperature': input_data.temperature, 'Sunlight_Intensity': input_data.sunlight_intensity,
                               'Humidity': input_data.humidity, 'Panel_Orientation': 0, 'Panel_Position': 0}, index=[0])

    results = []
    max_current = 0
    for orientation in range(0, 360, 10):
        for position in range(0, 90, 10):
            data_point_copy = data_point.copy()  # make a copy of the dataframe
            data_point_copy.loc[:, 'Panel_Orientation'] = orientation  # update value of Panel_Orientation
            data_point_copy.loc[:, 'Panel_Position'] = position  # update value of Panel_Position
            
            current = predict_current(data_point_copy)

            if current > max_current:
                max_current = current
                results = [{'Panel_Orientation': orientation, 'Panel_Position': position, 'Current_Generated': current}]
    
    return results
