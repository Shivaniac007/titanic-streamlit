# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained logistic regression model + scaler
bundle = joblib.load("log_reg_bundle.pkl")
model = bundle["model"]
scaler = bundle["scaler"]

st.title("🚢 Titanic Survival Prediction (Logistic Regression)")

st.write("Enter passenger details to predict survival probability:")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.slider("Fare", 0, 500, 50)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert inputs into dataframe with SAME feature names as training
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [1 if sex == "female" else 0],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked_Q": [1 if embarked == "Q" else 0],
    "Embarked_S": [1 if embarked == "S" else 0]
})

# Scale Age & Fare (same as training)
input_data[['Age', 'Fare']] = scaler.transform(input_data[['Age', 'Fare']])

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader(f"Prediction: {'🟢 Survived' if prediction==1 else '🔴 Did Not Survive'}")
    st.write(f"Survival Probability: {probability:.2f}")
