import streamlit as st
import numpy as np
import pickle

# Load saved models
models = {
    "K-Nearest Neighbors": pickle.load(open("knn_model.pkl", "rb")),
    "Support Vector Machine": pickle.load(open("svm_model.pkl", "rb")),
    "Decision Tree": pickle.load(open("dt_model.pkl", "rb")),
    "Random Forest": pickle.load(open("rf_model.pkl", "rb")),
}

st.title("💓 Heart Disease Prediction App")
st.markdown("Enter the patient information below and select a model to predict the heart disease risk.")

# Input fields
age = st.number_input("Age", 1, 120, 45)
sex = st.selectbox("Sex", [0, 1])  # 0 = Female, 1 = Male
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed, 3 = Reversible)", [1, 2, 3])

# Combine inputs
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# Select model
selected_model_name = st.selectbox("Choose Model", list(models.keys()))
model = models[selected_model_name]

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.subheader("Result:")
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")