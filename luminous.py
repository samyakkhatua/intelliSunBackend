import random
import json
import datetime

# Define the range of values for the sensor readings
temp_range = (15, 40)
humidity_range = (30, 80)
sunlight_range = (0, 100)
current_range = (0, 10)

# Define the range of values for panel position and orientation
panel_position_range = (0, 90)
panel_orientation_range = (0, 359)

# Create an empty list to store the generated data
data = []

# Generate 100 data points
for i in range(100):
    # Generate random values for the sensor readings
    temp = round(random.uniform(temp_range[0], temp_range[1]), 2)
    humidity = round(random.uniform(humidity_range[0], humidity_range[1]), 2)
    sunlight = round(random.uniform(sunlight_range[0], sunlight_range[1]), 2)
    current = round(random.uniform(current_range[0], current_range[1]), 2)

    # Generate random values for panel position and orientation
    panel_position = random.randint(panel_position_range[0], panel_position_range[1])
    panel_orientation = random.randint(panel_orientation_range[0], panel_orientation_range[1])

    # Get the current date and time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create a dictionary to store the data for this data point
    data_point = {
        "Timestamp": timestamp,
        "Temperature": temp,
        "Humidity": humidity,
        "Sunlight_Intensity": sunlight,
"Current_Generated": current,
        "Panel_Position": panel_position,
        "Panel_Orientation": panel_orientation
    }

    # Append the data point to the data list
    data.append(data_point)

# Convert the data list to JSON format and write it to a file
with open("datav2.json", "w") as f:
    json.dump(data, f)
