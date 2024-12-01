# import streamlit as st
# import joblib
# import numpy as np

# # Load the trained model
# model = joblib.load('loan_default_model.pkl')

# # Streamlit app UI
# st.title("Loan Default Prediciton")


# # Input field for he user
# loan_amount = st.number_input("Loan Amount", min_value=0, step=1000)
# annual_income = st.number_input("Annual Income", min_value=0, step=1000)
# dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, step=0.1)


# # Feature array
# features = np.array([[loan_amount, annual_income, dti]])


# # Prediction button
# if st.button("Predict"):
#     prediction = model.predict(features)
#     st.write("Prediction: ", "Default" if prediction[0] == 1 else "No Default")

import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature list
model = joblib.load("loan_default_model_reduced.pkl")
with open("top_features.pkl", "rb") as f:
    top_features = joblib.load(f)

# Load the label mapping
label_mapping = joblib.load("label_mapping.pkl")
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Input form
st.title("Loan Default Prediction")
st.write("Provide input for the following features:")

user_input = {}
for feature in top_features:
    user_input[feature] = st.text_input(f"Enter value for {feature}:", "0")

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input])

# Ensure column order matches the model
input_df = input_df[top_features]

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_label = reverse_label_mapping[prediction[0]]
    st.write(f"Prediction: {prediction_label}")
