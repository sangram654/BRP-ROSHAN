import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model + feature list
data = joblib.load("bike_model.pkl")
model = data["model"]
features = data["features"]

st.title("🚴 Bike Sharing Demand Prediction")

hr = st.slider("Hour", 0, 23)
mnth = st.slider("Month", 1, 12)
weekday = st.slider("Weekday", 0, 6)
temp = st.number_input("Temperature")
hum = st.number_input("Humidity")
windspeed = st.number_input("Wind Speed")

# Cyclic encoding
hr_sin = np.sin(2 * np.pi * hr / 24)
hr_cos = np.cos(2 * np.pi * hr / 24)

mnth_sin = np.sin(2 * np.pi * mnth / 12)
mnth_cos = np.cos(2 * np.pi * mnth / 12)

weekday_sin = np.sin(2 * np.pi * weekday / 7)
weekday_cos = np.cos(2 * np.pi * weekday / 7)

# Create dataframe properly aligned
input_dict = {
    'hr': hr,
    'mnth': mnth,
    'weekday': weekday,
    'temp': temp,
    'hum': hum,
    'windspeed': windspeed,
    'hr_sin': hr_sin,
    'hr_cos': hr_cos,
    'mnth_sin': mnth_sin,
    'mnth_cos': mnth_cos,
    'weekday_sin': weekday_sin,
    'weekday_cos': weekday_cos
}

input_df = pd.DataFrame([input_dict])

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Bike Count: {int(prediction[0])}")
