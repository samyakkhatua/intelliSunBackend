import math
import random
import pandas as pd
import csv

# Define constants and variables
SUNLIGHT_CONSTANT = 1361 # W/m^2
SOLAR_PANEL_AREA = 1 # m^2
SOLAR_PANEL_EFFICIENCY = 0.2 # 20% efficiency
TEMPERATURE_RANGE = (10, 40) # °C
HUMIDITY_RANGE = (10, 90) # %
PANEL_ORIENTATIONS = ['North-South', 'East-West']
PANEL_POSITIONS = ['Flat', 'Tilted']
num_samples = 1000
dew_point = 15 # °C
air_pressure = 101.3 # kPa

# Define functions to simulate temperature, sunlight intensity, and humidity
def simulate_temperature(time_of_day, season, location):
    # Implement a mathematical model to simulate temperature based on time of day, season, and location
    return random.uniform(*TEMPERATURE_RANGE)

def simulate_sunlight_intensity(time_of_day, season, location):
    # Implement a mathematical model to simulate sunlight intensity based on time of day, season, and location
    return SUNLIGHT_CONSTANT * math.sin((2 * math.pi * time_of_day) / 24) * math.sin((2 * math.pi * season) / 365)

def simulate_humidity(temperature, dew_point, air_pressure):
    # Implement a mathematical model to simulate humidity based on temperature, dew point, and air pressure
    return random.uniform(*HUMIDITY_RANGE)

# Define a function to simulate panel orientation and position
def simulate_panel_orientation_and_position():
    orientation = random.choice(PANEL_ORIENTATIONS)
    position = random.choice(PANEL_POSITIONS)
    return orientation, position

# Define a function to simulate current generated
def simulate_current_generated(temperature, sunlight_intensity, humidity, orientation, position):
    # Implement a mathematical model to simulate current generated based on temperature, sunlight intensity, humidity, orientation, and position
    return SOLAR_PANEL_AREA * SOLAR_PANEL_EFFICIENCY * sunlight_intensity

# Generate synthetic data
data = []
for i in range(num_samples):
    time_of_day = random.randint(0, 23)
    season = random.randint(0, 364)
    location = (random.uniform(-90, 90), random.uniform(-180, 180))
    temperature = simulate_temperature(time_of_day, season, location)
    sunlight_intensity = simulate_sunlight_intensity(time_of_day, season, location)
    humidity = simulate_humidity(temperature, dew_point, air_pressure)
    orientation, position = simulate_panel_orientation_and_position()
    current_generated = simulate_current_generated(temperature, sunlight_intensity, humidity, orientation, position)
    data.append([temperature, sunlight_intensity, humidity, orientation, position, current_generated])

# print(data)

with open('synthetic_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Temperature', 'Sunlight_Intensity', 'Humidity', 'Panel_Orientation', 'Panel_Position', 'Current_Generated'])
    writer.writerows(data)