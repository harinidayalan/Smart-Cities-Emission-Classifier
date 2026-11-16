
# Smart Cities: Real-Time Industrial Smoke & Factory Emission Classifier

## Project Overview
This project uses survey responses and machine learning to predict public support for industrial emission monitoring.

## Objectives
- Analyze survey responses.
- Apply a trained Decision Tree classification model.
- Predict public support categories.
- Display prediction results through a Streamlit dashboard.

## Technologies Used
- Python
- Pandas
- NumPy
- Streamlit
- Scikit-learn
- Joblib

## Project Files
- app.py - Main application
- requirements.txt - Required Python packages
- models/best_model.pkl - Trained machine learning model
- dataset/X_features_raw.csv - Input reference dataset

## Installation
pip install -r requirements.txt

## Run Application
streamlit run app.py

## Expected Output
The application accepts survey responses and predicts the corresponding support category.
