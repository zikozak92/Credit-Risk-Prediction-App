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

Upload a dataset (CSV or Excel) containing loan applications, and the model will predict the likelihood of default for each application.
""")

st.subheader("Expected Data Format")
st.write("Your dataset should contain the following columns:")

required_columns = [
    "loan_id", "person_age", "person_income", "person_home_ownership",
    "person_emp_length", "loan_intent", "loan_grade", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_default_on_file",
    "cb_person_cred_hist_length"
]

# Display columns in a table
st.table(pd.DataFrame({"Required Columns": required_columns}))

st.write("Here’s an example of how your data might look:")

# Sample data
sample_data = pd.DataFrame([
    [14668, 24, 28000, "OWN", 6, "HOMEIMPROVEMENT", "B", 10000, 10.37, 0.36, "N", 2],
    [24614, 27, 64000, "RENT", 0, "PERSONAL", "C", 10000, 15.27, 0.16, "Y", 10],
    [11096, 26, 72000, "MORTGAGE", 10, "EDUCATION", "D", 16000, None, 0.22, "N", 3],
    [10424, 23, 27996, "RENT", 7, "DEBTCONSOLIDATION", "A", 10000, None, 0.36, "N", 2]
], columns=required_columns)

st.dataframe(sample_data)

st.subheader("Load Data for Prediction")
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
