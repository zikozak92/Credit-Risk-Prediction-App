import streamlit as st
import pandas as pd
from src.preprocess import split_data, preprocess_data, format_classification_report
from src.visualization import plot_confusion_matrix, plot_roc_curves, plot_pr_curves
from src.models_func import *

# Custom CSS for enhanced styling
with open("static/style.css", "r") as f:
    (st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True))

st.title("Credit Risk Model Training & Performance")
st.markdown("This page demonstrates the dataset preprocessing, training, and evaluation of the credit risk model.")

# --- Load dataset directly ---
DATA_PATH = "data/credit_risk_dataset.csv"
df = pd.read_csv(DATA_PATH)
st.success("✅ Credit Risk Dataset loaded successfully!")

st.header("1️⃣ Data Splitting")
st.write(
    "We split the dataset into training and testing sets. "
    "The training set is used to train the model, and the testing set "
    "is used to evaluate its performance. Stratification ensures the "
    "class distribution is preserved in both sets."
)

# Perform the split
X_train, X_test, y_train, y_test = split_data(df)

st.write("**Training set size:**", X_train.shape)
st.write("**Testing set size:**", X_test.shape)

st.header("2️⃣ Preprocessing: Imputation & Encoding")
st.write(
    "We handle missing values and encode categorical features so the model "
    "can understand them. Numerical columns are imputed with the median, "
    "ordinal features use an ordinal encoder, and other categorical columns use one-hot encoding."
)

X_train_processed, X_test_processed, preprocessor = preprocess_data(X_train, X_test)

# all_feature_names = get_feature_names(preprocessor)

# X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names, index=X_train.index)
# X_test_df = pd.DataFrame(X_test_processed, columns=all_feature_names, index=X_test.index)

st.success("✅ Preprocessing complete.")

st.write("**Processed training set shape:**", X_train_processed.shape)
st.write("**Processed testing set shape:**", X_test_processed.shape)

st.subheader("Preview of Processed Training Data")
st.dataframe(X_train_processed.head(5))

st.header("3️⃣ Model Training & Comparison")
st.write(
    "We train three models to predict loan defaults: Random Forest, XGBoost, "
    "and LightGBM. All models use the best parameters from prior grid search. "
    "We evaluate models using Accuracy, Precision, Recall, F1-score, and ROC-AUC."
)

st.subheader("🌲 Random Forest")

rf_model = train_random_forest(X_train_processed, y_train)
rf_y_pred, rf_y_proba, rf_report, rf_auc = evaluate_model(rf_model, X_test_processed, y_test)

col1, col2 = st.columns([0.6, 0.4], gap='large')
with col1:
    st.write("Classification Report")
    st.dataframe(format_classification_report(rf_report))
    st.write(f"**ROC-AUC:** {rf_auc:.3f}")
with col2:
    st.write("Confusion Matrix")
    st.plotly_chart(plot_confusion_matrix(y_test, rf_y_pred))
# --- XGBoost ---
st.subheader("⚡ XGBoost")
xgb_model = train_xgboost(X_train_processed, y_train)
xgb_y_pred, xgb_y_proba, xgb_report, xgb_auc = evaluate_model(xgb_model, X_test_processed, y_test)

col1, col2 = st.columns([0.6, 0.4], gap='large')
with col1:
    st.write("Classification Report")
    st.dataframe(format_classification_report(xgb_report))
    st.write(f"**ROC-AUC:** {xgb_auc:.3f}")
with col2:
    st.write("Confusion Matrix")
    st.plotly_chart(plot_confusion_matrix(y_test, xgb_y_pred))


# --- LightGBM ---
st.subheader("💡 LightGBM")
lgbm_model = train_lightgbm(X_train_processed, y_train)
lgbm_y_pred, lgbm_y_proba, lgbm_report, lgbm_auc = evaluate_model(lgbm_model, X_test_processed, y_test)

col1, col2 = st.columns([0.6, 0.4], gap='large')
with col1:
    st.write("Classification Report")
    st.dataframe(format_classification_report(lgbm_report))
    st.write(f"**ROC-AUC:** {lgbm_auc:.3f}")
with col2:
    st.write("Confusion Matrix")
    st.plotly_chart(plot_confusion_matrix(y_test, lgbm_y_pred))

# --- ROC Curves ---
st.subheader("📈 Model Comparison")

# Dictionary of model probabilities
y_probas = {
    "Random Forest": rf_y_proba,
    "XGBoost": xgb_y_proba,
    "LightGBM": lgbm_y_proba
}

col1, col2 = st.columns([0.5, 0.5], gap='large')
with col1:
    st.write("**ROC-AUC Curves**")
    roc_fig = plot_roc_curves(y_test, y_probas)
    st.plotly_chart(roc_fig)
with col2:
    st.write("**Precision-Recall (PR) curves**")
    pr_fig = plot_pr_curves(y_test, y_probas)
    st.plotly_chart(pr_fig)


st.header("4️⃣ Conclusion")

st.markdown("""
After comparing the three models (Random Forest, XGBoost, and LightGBM), we observed the following:

- **Random Forest** achieved a solid ROC-AUC of **0.916**, with excellent recall for the majority class but lower recall for defaults.  
- **XGBoost** performed better, with a ROC-AUC of **0.948**, striking a more balanced trade-off between precision and recall.  
- **LightGBM** delivered the best overall performance, achieving a ROC-AUC of **0.955**. It demonstrated strong precision (**0.966**) for default predictions while maintaining very high accuracy (**93.6%**).

📊 Given its superior balance between predictive power and efficiency, we selected **LightGBM as the final model** for deployment in this app.  
This model was trained with optimized hyperparameters (via Grid Search) and saved as a reusable pipeline.  

In the **Prediction** page, you can upload new loan applications, and the app will:
- Predict the probability of default,  
- Provide an easy-to-understand classification (default / no default),  
- Offer explainability insights using SHAP values.  
""")

# # Footer
st.markdown("---")
st.markdown(" **Disclaimer:** This demo is for educational and portfolio purposes only.", unsafe_allow_html=True)
st.markdown(" Built with Streamlit by Zakaria Elyazghi | Data Source: [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data) | Last Updated: September 2025", unsafe_allow_html=True)
