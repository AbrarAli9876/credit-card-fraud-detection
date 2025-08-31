import streamlit as st
import pandas as pd
import numpy as np
import os
from model import predict_fraud # Import the prediction function from model.py

# --- Define Base Directory for Model Loading ---
# This ensures model.pkl is expected in the SAME directory as this script.
BASE_DIR_APP = os.path.dirname(os.path.abspath(__file__))
# --- End Base Directory Definition ---

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Credit Card Fraud Detector 💳",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling (subtle improvements) ---
st.markdown("""
    <style>
    .reportview-container .main {
        padding-top: 2rem;
    }
    .stButton>button {
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stSlider > div > div > div > div {
        background-color: #6a0572; /* A nice purple for sliders */
    }
    .stSlider label {
        font-weight: bold;
    }
    .st-emotion-cache-1r4qj8m {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .css-1cpxqw2 {
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title and Description ---
st.title("💳 Credit Card Fraud Detection App")
st.markdown("---")

st.markdown("""
    Welcome to the **Credit Card Fraud Detection App**!
    This tool helps identify potentially fraudulent credit card transactions
    using a machine learning model trained on a real-world dataset.

    Simply enter the transaction details below, and let our model do the heavy lifting!
    """, unsafe_allow_html=True)

# --- Check for model.pkl existence using the BASE_DIR_APP path ---
MODEL_PATH_APP = os.path.join(BASE_DIR_APP, 'model.pkl')
if not os.path.exists(MODEL_PATH_APP):
    st.error("🚨 **Model file `model.pkl` not found!**")
    st.markdown(f"""
        The model file is expected at: `{MODEL_PATH_APP}`.
        To get started, please train the machine learning model first.
        Open your terminal/command prompt, navigate to your project directory
        (`CREDIT CARD FRAUD DETECTION/`), and run:
        ```bash
        python train_model.py
        ```
        This script will train the model, save it as `model.pkl` in this directory,
        and also print its performance metrics. Once done, refresh this page.
    """)
    st.stop() # Stop the Streamlit app execution if model is not found

st.header("📋 Enter Transaction Details")
st.markdown("---")

# --- Input Fields for Transaction Features ---
col1, col2 = st.columns(2)

with col1:
    time = st.number_input(
        "⏱️ Transaction Time (minutes since first transaction)",
        min_value=0.0,
        value=2057.60,
        step=1.0,
        format="%.2f",
        help="Time in minutes elapsed from the very first transaction in the dataset. This feature helps the model understand transaction patterns over time."
    )
with col2:
    amount = st.number_input(
        "💰 Transaction Amount (₹)",
        min_value=0.0,
        value=7500.00,
        step=0.01,
        format="%.2f",
        help="The monetary value of the transaction. High amounts might sometimes be an indicator."
    )

st.markdown("---")
st.subheader("🕵️ Anonymized Features (V1-V28)")
st.markdown("""
    These features are the result of a **Principal Component Analysis (PCA)** transformation,
    provided due to confidentiality issues with the original transaction features.
    They represent complex patterns in the data. Adjusting these values will directly
    impact the model's prediction.
    """)

with st.expander("Adjust V Features (Advanced)"):
    v_features = {f'V{i}': 0.0 for i in range(1, 29)}
    selected_v_features_for_ui = [
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'
    ]

    cols_per_row = 4
    v_feature_columns = st.columns(cols_per_row)

    for idx, v_name in enumerate(selected_v_features_for_ui):
        with v_feature_columns[idx % cols_per_row]:
            v_features[v_name] = st.slider(
                f"**{v_name}**",
                min_value=-30.0,
                max_value=30.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                help=f"Anonymized principal component feature: {v_name}."
            )

transaction_data = {
    'Time': time,
    'Amount': amount,
    **v_features
}

# --- Prediction Button ---
st.markdown("---")
if st.button("🚀 Analyze Transaction", use_container_width=True, type="primary"):
    with st.spinner("Analyzing transaction for potential fraud... please wait ⏳"):
        try:
            prediction, probability = predict_fraud(transaction_data)

            st.markdown("---")
            st.subheader("💡 Prediction Result")
            if prediction == 1:
                st.error(f"🚨 **FRAUDULENT TRANSACTION DETECTED!**")
                st.markdown(f"""
                        The model predicts this transaction as **FRAUDULENT** with a
                        **confidence of {probability:.2%}**.
                        """)
                st.warning("Immediate action recommended: This transaction shows strong indicators of fraud. Please investigate and take appropriate measures.")
            else:
                st.success(f"✅ **Transaction is LIKELY LEGITIMATE.**")
                st.markdown(f"""
                        The model predicts this transaction as **LEGITIMATE** with a
                        **confidence of {(1 - probability):.2%}**.
                        """)
                st.info("This transaction appears to be normal based on the model's analysis. No immediate fraud indicators detected.")

            st.markdown("---")
            st.subheader("📊 Raw Prediction Data")
            st.markdown(f"**Predicted Class:** `{prediction}` (0 = Legitimate, 1 = Fraud)")
            st.markdown(f"**Probability of Fraud (Class 1):** `{probability:.2%}`")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Please ensure the model (`model.pkl`) is correctly trained and loaded. Check your terminal for more details.")

st.markdown("---")
st.markdown("""
    This application is designed for **demonstration and educational purposes only**. A robust, production-grade fraud detection system would typically involve more complex models, continuous data pipeline integration, and advanced alert mechanisms.
    """)
st.markdown("---")
# --- Updated Footer with Developer Information ---
st.markdown(f"""
    **Developed by:**
    * **Name:** K.S. Abrar Ali Ahmed
    * **Degree Pursuing:** Artificial Intelligence and Data Science
    * **College Name:** K S School Of Engineering and Management
    """)

# --- End Updated Footer ---
