# Credit Risk Prediction App

## Overview
This is a Streamlit web application that predicts the likelihood of loan default. It leverages machine learning models trained on real-world credit risk data to classify loan applications as “default” or “paid.”

## Features
- **Exploratory Data Analysis (EDA)**: Visualizations of target distribution, feature distributions, correlations, and default rates.
- **Model Training and Evaluation**: Compare Random Forest, XGBoost, and LightGBM models. Metrics include ROC-AUC, confusion matrices, and classification reports.
- **Prediction Interface**: Upload CSV or Excel files with loan applications to get predictions with probabilities.
- **Model Explanation**: SHAP waterfall plots explain feature contributions for each prediction.
- **Saved Pipeline**: The LightGBM model pipeline is saved for reuse on new unseen data.
