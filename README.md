# Global Air Quality Analysis and Prediction

## Project Overview

This project was completed as the final project of the Developer Academy Data Science Bootcamp. The aim was to analyse global air quality data, identify regions with the highest levels of pollution, and develop tools to predict air quality ratings using environmental and pollutant measurements.

The project combined exploratory data analysis, machine learning, environmental research, and application development to provide data-driven insights into air pollution trends around the world.

## Datasets

Two Kaggle datasets were used:

### Air Quality Dataset

* Approximately 5,000 regional observations
* Included environmental factors and pollutant concentrations
* Contained an air quality rating used as the target variable

### PM2.5 Dataset

* Approximately 6,985 cities
* Average annual PM2.5 concentrations
* Data covering the years 2017–2023

These datasets were combined to explore both pollution levels and long-term air quality trends.

## Exploratory Data Analysis

Extensive exploratory analysis was conducted to understand relationships between pollutants and air quality ratings.

Analysis included:

* Distribution analysis of pollutant concentrations
* Correlation analysis
* Air quality category comparisons
* Country and city-level pollution comparisons
* Trend analysis across multiple years
* Visualisation of global pollution patterns

## Air Quality Assessment Framework

A pollution assessment framework was developed using World Health Organization (WHO) air quality guidelines.

Threshold limits were established for key pollutants and compared against recorded measurements to identify areas exceeding recommended standards.

This allowed pollution levels to be interpreted using recognised environmental health benchmarks.

## Machine Learning Model

A predictive model was developed to estimate air quality ratings based on environmental and pollutant measurements.

The workflow included:

* Data cleaning and preprocessing
* Feature selection
* Model training and evaluation
* Prediction of air quality categories

The resulting model enabled air quality ratings to be generated from user-provided environmental measurements.

## PM2.5 Trend Analysis

Historical PM2.5 data from 2017–2023 was analysed to:

* Track pollution trends over time
* Identify cities with consistently high pollution levels
* Compare countries and regions
* Highlight locations experiencing worsening or improving air quality

The analysis identified several cities and countries with particularly severe pollution challenges.

## Streamlit Applications

Two interactive Streamlit applications were developed.

### Air Quality Prediction App

Users can input environmental and pollutant measurements and receive:

* Predicted air quality rating
* Pollution classification

### PM2.5 Lookup App

Users can search for a city and receive:

* Latest PM2.5 concentration
* Air pollution assessment
* Relevant air quality information

These applications transformed the analysis into user-friendly tools for non-technical users.

## Skills Demonstrated

* Python Programming
* Data Cleaning and Preparation
* Exploratory Data Analysis
* Statistical Analysis
* Environmental Data Analysis
* Machine Learning
* Feature Engineering
* Predictive Modelling
* Time-Series Trend Analysis
* Data Visualisation
* Streamlit Application Development
* Dashboard and User Interface Design

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Jupyter Notebook

## Conclusion

This project demonstrates the use of data science techniques to analyse and predict air quality using real-world environmental data. By combining machine learning, trend analysis, WHO guideline comparisons, and interactive Streamlit applications, the project provided both analytical insight and practical tools for exploring global air pollution.

