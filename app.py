import streamlit as st
import numpy as np
import joblib  # For loading the trained model
import pandas as pd

def load_model():
    return joblib.load("air_quality_model.pkl")  # Replace with your actual model file

# Load the trained model
model = load_model()

# Streamlit App
st.title("Air Quality Prediction App")
st.write("Enter the required air quality variables to predict the air quality level.")

# Load dataset to get feature names (assuming CSV structure is known)
df = pd.read_csv("aq-5000.csv")  # Ensure the correct path or provide a sample structure
feature_columns = df.columns[:-1]  # Assuming last column is the target variable

# Define custom labels with units and descriptions
feature_labels = {
    "Temperature": "Temperature (°C)",
    "Humidity": "Humidity (%)",
    "PM2.5": "PM2.5 Concentration (µg/m³)",
    "PM10": "PM10 Concentration (µg/m³)",
    "NO2": "NO2 Concentration (ppb)",
    "SO2": "SO2 Concentration (ppb)",
    "CO": "CO Concentration (ppm)",
    "Proximity_to_Industrial_Areas": "Distance to the nearest industrial zone (km)",
    "Population_Density": "Population Density (people/km²)"
}

# Dynamic user inputs based on dataset features
user_inputs = []
for feature in feature_columns:
    label = feature_labels.get(feature, feature)  # Use label if available, else default to feature name
    value = st.number_input(f"{label}", min_value=0.0, step=0.1)
    user_inputs.append(value)

# Prediction
if st.button("Predict Air Quality"):
    features = np.array([user_inputs])  # Ensure correct feature order
    prediction = model.predict(features)[0]  # Get prediction
    
    # Mapping prediction to labels (Merging Poor & Hazardous)
    label_map = {0: "Good", 1: "Moderate", 2: "Poor"}  # 2 includes both Poor & Hazardous
    air_quality = label_map.get(prediction, "Unknown")
    
    st.write(f"Predicted Air Quality Level: **{air_quality}**")