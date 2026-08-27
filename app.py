import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load("loan_model.pkl")

st.title("🏦 Loan Approval Prediction System")
st.write("Enter applicant details below to check loan eligibility:")

# Input fields for user

income_annum = st.number_input("Annual Income (in ₹)", min_value=0, value=500000)
loan_amount = st.number_input("Loan Amount Requested (in ₹)", min_value=0, value=1000000)
loan_term = st.number_input("Loan Term (in Years)", min_value=1, max_value=30, value=10)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)
residential_assets_value = st.number_input("Residential Asset Value (in ₹)", min_value=0, value=1000000)
commercial_assets_value = st.number_input("Commercial Asset Value (in ₹)", min_value=0, value=0)
luxury_assets_value = st.number_input("Luxury Asset Value (in ₹)", min_value=0, value=200000)
bank_asset_value = st.number_input("Bank Asset Value (in ₹)", min_value=0, value=300000)

if st.button("🚀 Predict Loan Status"):
    features = np.array([[ income_annum, loan_amount, loan_term, 
                          cibil_score, residential_assets_value, commercial_assets_value, 
                          luxury_assets_value, bank_asset_value]])
    
    prediction = model.predict(features)
    
    if prediction[0] == 0:
        st.success("🎉 Congratulations! Loan Approved!")
    else:
        st.error("❌ Sorry, Loan Rejected based on evaluation parameters.")
