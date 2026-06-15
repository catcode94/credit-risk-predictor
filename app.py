# 1 Good (lower risk) and 0 Bad (Higher risk)

import streamlit as st
import pandas as pd
import joblib

# load model and encoder using joblib
model = joblib.load("xgb_credit_model.pkl")
encoders = {col : joblib.load(f"{col}_encoder.pkl") for col in ["Sex", "Saving accounts", "Checking account"]}

st.title("Credit Risk Prediction App")
st.write("Enter applicant information to predict if the credit risk is good or bad")

age = st.number_input("Age", min_value =18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male","female"])
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"])
checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"])
credit_amount = st.number_input("Credit Amount", min_value=100, value=5000)
duration = st.number_input("Duration (months)", min_value=1, value=12)
# input_df = pd.DataFrame({
#     "Age": [age],
#     "Sex": [encoders["Sex"].transform([sex])[0]],
#     "Job": [job],
#     "Housing_own": [1 if housing == "own" else 0],
#     "Housing_rent": [1 if housing == "rent" else 0],
#     "Saving accounts": [encoders["Saving accounts"].transform({saving_accounts}) [0]],
#     "Checking account": [encoders["Checking account"].transform({checking_account})[0]],
#     "Credit amount": [credit_amount],
#     "Duration": [duration]
# })


input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration],
    "Housing_own": [1 if housing == "own" else 0],
    "Housing_rent": [1 if housing == "rent" else 0]
})

if st.button("Predict Risk"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("The predicted credit risk score is **GOOD** (lower risk).")

    else:
        st.error("The predicted credit risk score is **BAD** (higher risk).")