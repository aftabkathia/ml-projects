import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('rf_best_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Cardiotocography (CTG) Health Prediction")
st.write("Adjust the sliders below to predict the Fetal State (NSP).")

# Create sliders for the 18 features used in training
st.sidebar.header("Input Patient Data")

def user_input_features():
    LB = st.sidebar.slider('Baseline Fetal Heart Rate (LB)', 106.0, 160.0, 133.0)
    AC = st.sidebar.slider('Accelerations (AC)', 0.0, 26.0, 2.0)
    FM = st.sidebar.slider('Fetal Movement (FM)', 0.0, 564.0, 10.0)
    UC = st.sidebar.slider('Uterine Contractions (UC)', 0.0, 16.0, 4.0)
    ASTV = st.sidebar.slider('% Time with Abnormal Short Term Variability (ASTV)', 12.0, 87.0, 47.0)
    MSTV = st.sidebar.slider('Mean Short Term Variability (MSTV)', 0.2, 7.0, 1.3)
    ALTV = st.sidebar.slider('% Time with Abnormal Long Term Variability (ALTV)', 0.0, 91.0, 10.0)
    MLTV = st.sidebar.slider('Mean Long Term Variability (MLTV)', 0.0, 50.0, 8.0)
    Width = st.sidebar.slider('Width of FHR Histogram', 3.0, 180.0, 70.0)
    Min = st.sidebar.slider('Minimum of FHR Histogram', 50.0, 159.0, 93.0)
    Max = st.sidebar.slider('Maximum of FHR Histogram', 122.0, 238.0, 164.0)
    Nmax = st.sidebar.slider('Number of Histogram Peaks', 0.0, 18.0, 4.0)
    Nzeros = st.sidebar.slider('Number of Histogram Zeros', 0.0, 10.0, 0.0)
    Mode = st.sidebar.slider('Histogram Mode', 60.0, 187.0, 137.0)
    Mean = st.sidebar.slider('Histogram Mean', 73.0, 182.0, 134.0)
    Median = st.sidebar.slider('Histogram Median', 77.0, 186.0, 138.0)
    Variance = st.sidebar.slider('Histogram Variance', 0.0, 269.0, 18.0)
    Tendency = st.sidebar.slider('Histogram Tendency', -1.0, 1.0, 0.0)
    
    data = {'LB': LB, 'AC': AC, 'FM': FM, 'UC': UC, 'ASTV': ASTV, 'MSTV': MSTV,
            'ALTV': ALTV, 'MLTV': MLTV, 'Width': Width, 'Min': Min, 'Max': Max,
            'Nmax': Nmax, 'Nzeros': Nzeros, 'Mode': Mode, 'Mean': Mean,
            'Median': Median, 'Variance': Variance, 'Tendency': Tendency}
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display the inputs
st.subheader('User Input parameters')
st.write(input_df)

# Scaling and Prediction
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)

# Map prediction back to labels
labels = {1.0: "Normal", 2.0: "Suspect", 3.0: "Pathologic"}
result = labels.get(prediction[0], "Unknown")

st.subheader('Prediction Result')
st.success(f"The predicted Fetal State is: **{result}**")