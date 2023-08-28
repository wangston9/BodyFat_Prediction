import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

import streamlit as st

def main():
    st.title("Body Fat Prediction App")

    st.write("Welcome to the Body Fat Prediction App! This app predicts body fat using a machine learning model.")

    st.markdown("Data Source: [Body Fat Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset)")

    st.markdown("The data were generously supplied by Dr. A. Garth Fisher who gave permission to freely distribute the data and use for non-commercial purposes.")

if __name__ == "__main__":
    main()

# Image
st.image("BodyFatPercentageChart.jpg", width=700)

# Load the cleaned data from the CSV file
data = pd.read_csv('cleaned_BodyFat.csv', index_col=0, encoding= 'unicode_escape')

# Example: Display the first few rows of the loaded data
st.write("Shape of Dataset:", data.shape)
st.write("Sample of Dataset:")
st.write(data.head())

st.sidebar.header('User Input Parameters')

def user_input_features(): 
    Density = st.sidebar.slider('Density from UnderWater Weighing', 0.99, 1.2, 1.05)
    Age = st.sidebar.slider('Age', 0, 100, 54)
    Weight = st.sidebar.slider('Weight(kg)', 20, 112, 60)
    Height = st.sidebar.slider('Height(cm)', 130, 200, 167)
    Neck = st.sidebar.slider('Neck(cm)', 25, 50, 40)
    Chest = st.sidebar.slider('Chest(cm)', 60, 122, 80)
    Abdomen = st.sidebar.slider('Abdomen(cm)',60, 120, 75)
    Hip = st.sidebar.slider('Hip(cm)',80, 120, 99)
    Thigh = st.sidebar.slider('Thigh(cm)', 40, 75, 60)
    Knee = st.sidebar.slider('Knee(cm)', 25, 50, 34)   
    Ankle = st.sidebar.slider('Ankle(cm)', 15, 45, 28)
    Biceps = st.sidebar.slider('Biceps(cm)', 20, 45, 38)
    Forearm = st.sidebar.slider('Forearm(cm)', 20, 45, 40)
    Wrist = st.sidebar.slider('Wrist(cm)', 10, 25, 18)
    data = {'Density': Density,
            'Age': Age,
            'Weight': Weight,
            'Height': Height,
            'Neck': Neck,
            'Chest': Chest,
            'Abdomen': Abdomen,
            'Hip': Hip,
            'Thigh': Thigh,
            'Knee': Knee,
            'Ankle': Ankle,
            'Biceps': Biceps,
            'Forearm': Forearm,
            'Wrist': Wrist}
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

if st.checkbox("Show Statistics"):
    st.table(data.describe())

# Load the trained model
trained_model = joblib.load('linear_regression.pkl')

# Create a button to make predictions
if st.button("Predict Body Fat (Linear Regression)"):
    if trained_model is None:
        st.warning("Please load the trained model first.")
    else:
        # Prepare the input data for prediction
        input_data = df

        # Make the prediction using the loaded trained model
        prediction = trained_model.predict(input_data)

        # Display the prediction result
        st.write(f"Predicted Body Fat Percentage: {prediction[0]:.2f}%")

