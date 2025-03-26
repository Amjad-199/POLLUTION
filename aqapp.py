import streamlit as st
import pandas as pd

# Load the data
df = pd.read_csv('pm2.5.csv')

# Drop rows with missing PM2.5 in 2023
df = df.dropna(subset=['2023'])

# Streamlit UI
st.title("PM2.5 Air Quality Checker 🌍")

# User input for city
city = st.text_input("Enter a city name:")

if city:
    result = df[df['city'].str.lower() == city.lower()]
    
    if not result.empty:
        pm25 = result['2023'].values[0]
        
        # Define categories
        if pm25 <= 15:
            category = "Good Air Quality (≤ 15 μg/m³)"
            st.success("✅ Good Air Quality!")
        elif 15 < pm25 <= 30:
            category = "Moderate Pollution (15 - 30 μg/m³)"
            st.warning("⚠️ Moderate Pollution Level!")
        else:
            category = "High Pollution (> 30 μg/m³)"
            st.error("🚨 High Pollution Level!")
        
        # Display results
        st.write(f"**City:** {result['city'].values[0]}, **Country:** {result['country'].values[0]}")
        st.write(f"**Latest PM2.5:** {pm25:.2f} μg/m³")
        st.write(f"**Category:** {category}")
    
    else:
        st.warning("City not found. Please try another city.")
