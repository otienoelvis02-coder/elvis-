[streamlit_app.py](https://github.com/user-attachments/files/24736657/streamlit_app.py)
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Loan Default Predictor", layout="wide")

st.title("🏦 Loan Default Prediction App")
st.markdown("---")

# Sidebar for information
st.sidebar.header("About")
st.sidebar.info(
    """
    This app predicts the likelihood of loan default based on 
    loan characteristics using a Machine Learning model.
    """
)

# Main content
st.markdown("""
### Predict Loan Default Risk
Enter the loan details below to predict if the loan will default.
""")

# Create columns for input
col1, col2, col3 = st.columns(3)

with col1:
    disbursed_amount = st.number_input("Disbursed Amount ($)", min_value=0.0, step=1000.0)
    interest = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.5)
    number_of_payments = st.number_input("Number of Payments", min_value=1, step=1)

with col2:
    employment_years = st.number_input("Employment Years", min_value=0, step=1)
    income = st.number_input("Annual Income ($)", min_value=0.0, step=1000.0)
    debt_to_income = st.number_input("Debt to Income Ratio", min_value=0.0, step=0.1)

with col3:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=10)
    loan_age_months = st.number_input("Loan Age (months)", min_value=0, step=1)
    payment_history = st.slider("Payment History Score (0-100)", 0, 100, 50)

# Prediction button
if st.button("🔮 Predict Loan Default", type="primary", use_container_width=True):
    
    # Prepare input data
    input_data = np.array([[
        disbursed_amount,
        interest,
        number_of_payments,
        employment_years,
        income,
        debt_to_income,
        credit_score,
        loan_age_months,
        payment_history
    ]])
    
    # Normalize input (same as training)
    scaler = MinMaxScaler()
    input_normalized = scaler.fit_transform(input_data)
    
    # Simple prediction using LogisticRegression as example
    # In production, load your trained model
    model = LogisticRegression(random_state=42)
    
    # Generate prediction
    prediction_prob = np.random.random()  # Placeholder - replace with actual model
    prediction = 1 if prediction_prob > 0.5 else 0
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 0:
            st.success("✅ Low Default Risk", icon="✅")
            risk_level = "Low"
        else:
            st.error("⚠️ High Default Risk", icon="⚠️")
            risk_level = "High"
    
    with col2:
        st.metric("Default Probability", f"{prediction_prob*100:.2f}%")
    
    # Summary
    st.markdown("---")
    st.subheader("📋 Loan Summary")
    summary_data = {
        "Loan Amount": f"${disbursed_amount:,.2f}",
        "Interest Rate": f"{interest}%",
        "Credit Score": f"{credit_score}",
        "Risk Assessment": risk_level,
        "Confidence": f"{max(prediction_prob, 1-prediction_prob)*100:.2f}%"
    }
    
    for key, value in summary_data.items():
        st.write(f"**{key}:** {value}"
