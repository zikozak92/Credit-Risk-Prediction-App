import streamlit as st
import pandas as pd
from src import visualization as viz

# Custom CSS for enhanced styling
with open("static/style.css", "r") as f:
    (st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True))

st.set_page_config(page_title="Credit Risk EDA", layout="wide")

st.title("Exploratory Data Analysis (EDA) - Credit Risk Dataset")
st.markdown("""
This page shows interactive plots and insights about the credit risk dataset.
Explore target distribution, missing values, numerical and categorical features, and correlations.
""")

# --- Load dataset directly ---
DATA_PATH = "data/credit_risk_dataset.csv"
df = pd.read_csv(DATA_PATH)
st.success("✅ Credit Risk Dataset loaded successfully!")
st.dataframe(df.head())

# --- Identify columns ---
target = "loan_status"
num_cols = ["person_age", "person_income", "person_emp_length",
            "loan_amnt", "loan_int_rate", "loan_percent_income"]
cat_cols = df.select_dtypes(include=['object']).columns.tolist()


# --- Target Distribution ---
st.subheader("Target Distribution")
st.plotly_chart(viz.plot_target_distribution(df, target))

# --- Missing Values ---
# st.subheader("Missing Values")
# missing_fig = viz.plot_missing_values(df)
# if missing_fig:
#     st.plotly_chart(missing_fig)
# else:
#     st.info("No missing values!")

# --- Numerical Features Distribution ---
st.subheader("Numerical Features Distribution")
figs = viz.plot_numerical_distributions(df, num_cols)
# Display 3 plots per row
for i in range(0, len(figs), 3):
    cols = st.columns(3)
    for j, fig in enumerate(figs[i:i+3]):
        cols[j].plotly_chart(fig, use_container_width=True)

# --- Numerical Features vs Target ---
st.subheader("Numerical Features vs Target")
num_target_figs = viz.plot_numerical_vs_target(df, num_cols[:6], target)
# Display 3 plots per row
for i in range(0, len(num_target_figs), 3):
    cols = st.columns(3)
    for j, fig in enumerate(num_target_figs[i:i+3]):
        cols[j].plotly_chart(fig, use_container_width=True)


# --- Categorical Features vs Target (Default Rate) ---
st.subheader("Categorical Features - Default Rate")
cat_target_figs = viz.plot_categorical_distributions(df, cat_cols, target)
# Display 3 plots per row
for i in range(0, len(cat_target_figs), 2):
    cols = st.columns(2)
    for j, fig in enumerate(cat_target_figs[i:i+2]):
        cols[j].plotly_chart(fig, use_container_width=True)

# --- Correlation Heatmap ---
st.subheader("Correlation Heatmap")
st.plotly_chart(viz.plot_correlation_heatmap(df, num_cols))

# # Footer
st.markdown("---")
st.markdown(" **Disclaimer:** This demo is for educational and portfolio purposes only.", unsafe_allow_html=True)
st.markdown(" Built with Streamlit by Zakaria Elyazghi | Data Source: [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data) | Last Updated: September 2025", unsafe_allow_html=True)
