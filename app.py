import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved models
models = {
    "K-Nearest Neighbors": pickle.load(open("knn_model.pkl", "rb")),
    "Support Vector Machine": pickle.load(open("svm_model.pkl", "rb")),
    "Decision Tree": pickle.load(open("dt_model.pkl", "rb")),
    "Random Forest": pickle.load(open("rf_model.pkl", "rb")),
}

# Load training feature names
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

st.title("💓 Heart Disease Prediction App")
st.markdown("Enter the patient information below and select a model to predict the heart disease risk.")

# Input fields
age = st.number_input("Age", 1, 120, 45)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed, 3 = Reversible)", [1, 2, 3])

# Prepare input dictionary
input_dict = {
    'age': age,
    'sex': int(sex),
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': int(fbs),
    'restecg': restecg,
    'thalach': thalach,
    'exang': int(exang),
    'oldpeak': oldpeak,
    'slope': int(slope),
    'ca': int(ca),
    'thal': thal
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# One-hot encode like training
input_df_encoded = pd.get_dummies(input_df, columns=['cp', 'restecg', 'thal'], drop_first=True)

# Add missing columns and reorder to match training
for col in feature_names:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0

input_df_encoded = input_df_encoded[feature_names]

# Model selection
selected_model_name = st.selectbox("Choose Model", list(models.keys()))
model = models[selected_model_name]

# Predict and display result
if st.button("Predict"):
    prediction = model.predict(input_df_encoded)[0]
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
