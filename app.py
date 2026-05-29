import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")
st.write("Predict the likelihood of heart disease using patient health data.")

# Load model and encoders
model = joblib.load("model/heart_model.pkl")
encoders = joblib.load("model/encoders.pkl")

# User Inputs

age = st.number_input("Age", 20, 100, 50)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

chest_pain_type = st.selectbox(
    "Chest Pain Type",
    [
        "Typical angina",
        "Atypical angina",
        "Non-anginal pain",
        "Asymptomatic"
    ]
)

resting_blood_pressure = st.number_input(
    "Resting Blood Pressure",
    80,
    250,
    120
)

cholestoral = st.number_input(
    "Cholesterol",
    100,
    600,
    200
)

fasting_blood_sugar = st.selectbox(
    "Fasting Blood Sugar",
    [
        "Lower than 120 mg/ml",
        "Greater than 120 mg/ml"
    ]
)

rest_ecg = st.selectbox(
    "Rest ECG",
    [
        "Normal",
        "ST-T wave abnormality",
        "Left ventricular hypertrophy"
    ]
)

Max_heart_rate = st.number_input(
    "Maximum Heart Rate",
    60,
    250,
    150
)

exercise_induced_angina = st.selectbox(
    "Exercise Induced Angina",
    [
        "Yes",
        "No"
    ]
)

oldpeak = st.number_input(
    "Oldpeak",
    0.0,
    10.0,
    1.0
)

slope = st.selectbox(
    "Slope",
    [
        "Upsloping",
        "Flat",
        "Downsloping"
    ]
)

vessels_colored_by_flourosopy = st.selectbox(
    "Colored Vessels",
    [
        "Zero",
        "One",
        "Two",
        "Three"
    ]
)

thalassemia = st.selectbox(
    "Thalassemia",
    [
        "Normal",
        "Fixed Defect",
        "Reversable Defect",
        "No"
    ]
)

if st.button("Predict Disease Risk"):

    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "resting_blood_pressure": resting_blood_pressure,
        "cholestoral": cholestoral,
        "fasting_blood_sugar": fasting_blood_sugar,
        "rest_ecg": rest_ecg,
        "Max_heart_rate": Max_heart_rate,
        "exercise_induced_angina": exercise_induced_angina,
        "oldpeak": oldpeak,
        "slope": slope,
        "vessels_colored_by_flourosopy": vessels_colored_by_flourosopy,
        "thalassemia": thalassemia
    }])

    # Encode categorical columns using saved encoders

    for col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if probability < 0.30:
        st.success(f"🟢 Low Risk ({probability:.2%})")

    elif probability < 0.70:
        st.warning(f"🟡 Medium Risk ({probability:.2%})")

    else:
        st.error(f"🔴 High Risk ({probability:.2%})")

    st.progress(float(probability))

    st.write(f"Risk Probability: **{probability:.2%}**")

    if prediction == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")