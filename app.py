import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Credit Card Customer Segmentation", layout="centered")

st.title("Credit Card Customer Segmentation")
st.caption("Enter a customer's credit card usage to find their segment")

base_dir = os.path.dirname(__file__)
scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
kmeans = joblib.load(os.path.join(base_dir, "kmeans_model.pkl"))
feature_order = joblib.load(os.path.join(base_dir, "feature_order.pkl"))

cluster_labels = {
    0: "Moderate balance, regular purchases, pays in full often",
    1: "Moderate balance, low purchases, rarely pays in full",
    2: "High purchases and payments, high credit limit",
    3: "Low balance, light spender",
    4: "High balance, heavy cash advance user",
}

with st.form("predict_form"):
    balance = st.number_input("Balance", min_value=0.0, value=1000.0)
    balance_frequency = st.slider("Balance Frequency", 0.0, 1.0, 0.9)
    purchases = st.number_input("Purchases", min_value=0.0, value=500.0)
    oneoff_purchases = st.number_input("One-off Purchases", min_value=0.0, value=200.0)
    installments_purchases = st.number_input("Installments Purchases", min_value=0.0, value=300.0)
    cash_advance = st.number_input("Cash Advance", min_value=0.0, value=0.0)
    purchases_frequency = st.slider("Purchases Frequency", 0.0, 1.0, 0.5)
    oneoff_purchases_frequency = st.slider("One-off Purchases Frequency", 0.0, 1.0, 0.2)
    purchases_installments_frequency = st.slider("Purchases Installments Frequency", 0.0, 1.0, 0.3)
    cash_advance_frequency = st.slider("Cash Advance Frequency", 0.0, 1.0, 0.0)
    cash_advance_trx = st.number_input("Cash Advance Transactions", min_value=0, value=0)
    purchases_trx = st.number_input("Purchases Transactions", min_value=0, value=10)
    credit_limit = st.number_input("Credit Limit", min_value=0.0, value=4000.0)
    payments = st.number_input("Payments", min_value=0.0, value=1500.0)
    minimum_payments = st.number_input("Minimum Payments", min_value=0.0, value=500.0)
    prc_full_payment = st.slider("Percent Full Payment", 0.0, 1.0, 0.1)
    tenure = st.number_input("Tenure (months)", min_value=6, max_value=12, value=12)
    submitted = st.form_submit_button("Calculate cluster")

if submitted:
    row = pd.DataFrame([{
        "BALANCE": balance,
        "BALANCE_FREQUENCY": balance_frequency,
        "PURCHASES": purchases,
        "ONEOFF_PURCHASES": oneoff_purchases,
        "INSTALLMENTS_PURCHASES": installments_purchases,
        "CASH_ADVANCE": cash_advance,
        "PURCHASES_FREQUENCY": purchases_frequency,
        "ONEOFF_PURCHASES_FREQUENCY": oneoff_purchases_frequency,
        "PURCHASES_INSTALLMENTS_FREQUENCY": purchases_installments_frequency,
        "CASH_ADVANCE_FREQUENCY": cash_advance_frequency,
        "CASH_ADVANCE_TRX": cash_advance_trx,
        "PURCHASES_TRX": purchases_trx,
        "CREDIT_LIMIT": credit_limit,
        "PAYMENTS": payments,
        "MINIMUM_PAYMENTS": minimum_payments,
        "PRC_FULL_PAYMENT": prc_full_payment,
        "TENURE": tenure,
    }])[feature_order]

    scaled_row = scaler.transform(row)
    predicted_cluster = int(kmeans.predict(scaled_row)[0])

    st.success(f"Cluster {predicted_cluster}")
    st.write(cluster_labels.get(predicted_cluster, ""))
