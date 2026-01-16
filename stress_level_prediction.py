import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# save file path to a variable for easier access
file_path = "synthetic_mental_health_dataset.csv"

# Read the data and store in dataframe called mental_health_dataset
mental_health_dataset = pd.read_csv(file_path)

# Choosing Features
features = ['sleep_hours', 'screen_time', 'exercise_minutes', 'daily_pending_tasks', 'interruptions', 'fatigue_level', 'social_hours', 'coffee_cups']
X = mental_health_dataset[features]

# Prediction Target
y = mental_health_dataset.stress_level

# splitting data into training data and testing data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define model
stress_level_model = RandomForestRegressor(random_state = 1)

# Fit model
stress_level_model.fit(train_X, train_y)

# Ask for user input
sleep_hours = float(input("Enter sleep hours per day: "))
screen_time = float(input("Enter screen Time(hours per day): "))
exercise_minutes = float(input("Enter Exercise Time(minutes): "))
daily_pending_tasks = int(input("Enter daily pending tasks: "))
interruptions = int(input("Enter interruptions: "))
fatigue_level = float(input("Enter Fatigue Level: "))
social_hours = float(input("Enter social hours: "))
coffee_cups = int(input("Enter coffee cups taken daily: "))

# Formatting input correctly
user_input = [[sleep_hours, screen_time, exercise_minutes, daily_pending_tasks, interruptions, fatigue_level, social_hours, coffee_cups]]

# Make the prediction
prediction = stress_level_model.predict(user_input)

# Output the predicted value
print("Predicted Stress Level: ", prediction[0])

# Getting the Mean Absolute error
stress_preds = stress_level_model.predict(val_X)
print("Mean Absolute Error: ", mean_absolute_error(val_y, stress_preds))

