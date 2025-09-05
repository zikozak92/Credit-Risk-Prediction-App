import streamlit as st
import pandas as pd
import os
import io
from src.preprocess import load_data, load_pipline, make_predictions, explain_predictions

# Custom CSS for enhanced styling
with open("static/style.css", "r") as f:
    (st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True))

# Streamlit app layout
st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

# --- Title and Introduction ---
st.title("Credit Risk Prediction App")
st.write("""
Welcome to the **Credit Risk Prediction App**.  
Upload a dataset (CSV or Excel) containing loan applications, and the model will help predict the likelihood of default.  
""")

# --- Checkbox for sample data ---
use_sample = st.checkbox("👉 Use sample dataset instead")

df = None

if use_sample:
    sample_path = os.path.join("data", "sample_data.csv")  # adjust file name
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)

        # Handle loan_id if present
        if "loan_id" in df.columns:
            df.set_index("loan_id", inplace=True)

        st.success("✅ Sample dataset loaded successfully!")
    else:
        st.error("⚠️ No sample file found in `data/` folder. Please upload your own file.")
else:
    uploaded_file = st.file_uploader("📂 Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.success("✅ File uploaded successfully!")

if df is not None:

    loaded_pipeline = load_pipline()

    results = make_predictions(df, loaded_pipeline)

    # Convert results DataFrame to CSV in memory
    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name="credit_risk_predictions.csv",
        mime="text/csv",
    )

    explain_predictions(df, loaded_pipeline)


# # Footer
st.markdown("---")
st.markdown(" **Disclaimer:** This demo is for educational and portfolio purposes only.", unsafe_allow_html=True)
st.markdown(" Built with Streamlit by Zakaria Elyazghi | Data Source: [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data) | Last Updated: September 2025", unsafe_allow_html=True)
