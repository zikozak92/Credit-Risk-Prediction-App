import pandas as pd
import streamlit as st
import os
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer


def load_data(file):
    """Load CSV or Excel file into a DataFrame"""
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # Set loan_id as index if present
        if "loan_id" in df.columns:
            df.set_index("loan_id", inplace=True)

        return df
    except Exception as e:
        st.error(f"⚠️ Error while reading file: {e}")
        return None


def load_pipline():
    pipeline_path = os.path.join("data", "credit_risk_pipeline.pkl")
    return joblib.load(pipeline_path)


def make_predictions(df, pipeline, threshold=0.5):
    """
    Run predictions on new loan applications using a saved pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with loan applications (must have same schema as training data).
    pipeline : str
        Path to the saved pipeline (joblib file).
    threshold : float
        Decision threshold for classification.

    Returns
    -------
    results : pd.DataFrame
        DataFrame with loan_id (index), predicted class, and probability of default.
    """

    # Drop target (and other unwanted) columns if they exist
    X_new = df.copy()

    # Predict probabilities and labels
    proba = pipeline.predict_proba(X_new)[:, 1]
    preds = (proba >= threshold).astype(int)

    # Return results
    results = pd.DataFrame({
        "Application_ID": X_new.index,
        "Predicted_Default": preds,
        "Probability_Default": proba
    }).set_index("Application_ID")

    return results


def explain_predictions(df, pipeline, threshold=0.5):
    """
    Predict + SHAP explanations with styled expanders for each application.
    """
    # Prepare input
    X_new = df.copy()

    # Predict
    proba = pipeline.predict_proba(X_new)[:, 1]
    preds = (proba >= threshold).astype(int)

    # Get model + preprocessor
    model = pipeline.named_steps["model"]       # update if different name
    preprocessor = pipeline.named_steps["preprocessor"]
    X_trans = preprocessor.transform(X_new)
    feature_names = preprocessor.get_feature_names_out()

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    # Loop through applications
    for i, idx in enumerate(X_new.index):
        prob = proba[i]
        label = "🔴" if preds[i] == 1 else "🟢"

        with st.expander(f"{label} Application {idx} | Probability of Default: {prob:.2f} "):

            # Mini table of original data
            st.write("### 📝 Application Data")
            st.dataframe(df.loc[[idx]].T, height=200)

            col1, col2 = st.columns(2, border=1)

            with col1:
                # SHAP Waterfall
                st.write("### 🔎 SHAP Waterfall (Top Features)")
                shap_expl = shap.Explanation(
                    values=shap_values[i],
                    base_values=explainer.expected_value,
                    data=X_trans[i],
                    feature_names=feature_names
                )
                fig, ax = plt.subplots(figsize=(6, 3))  # smaller plot
                shap.plots.waterfall(shap_expl, max_display=8, show=False)
                st.pyplot(fig)

            with col2:
                # SHAP Bar Plot
                st.write("### 📊 Feature Impact (Top 7)")
                shap_df = pd.DataFrame({
                    "feature": feature_names,
                    "value": X_trans[i],
                    "shap": shap_values[i]
                }).sort_values("shap", key=abs, ascending=False).head(7)

                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x="shap", y="feature", data=shap_df, palette="coolwarm", ax=ax)
                ax.set_title("Top SHAP Contributions")
                st.pyplot(fig)


def split_data(df, target_col="loan_status", test_size=0.2, random_state=42):
    """
    Split DataFrame into train and test sets.
    Returns X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test, ordinal_cols=None):
    """
    Impute missing values and encode categorical features.
    Returns transformed train/test DataFrames and fitted preprocessor.
    """
    if ordinal_cols is None:
        ordinal_cols = []

    # Identify columns
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    ohe_cols = [c for c in cat_cols if c not in ordinal_cols]
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # --- Pipelines ---
    numeric_pipeline = SimpleImputer(strategy='median')

    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OrdinalEncoder())
    ]) if ordinal_cols else 'passthrough'

    ohe_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]) if ohe_cols else 'passthrough'

    # ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, num_cols),
        ('ord', ordinal_pipeline, ordinal_cols),
        ('ohe', ohe_pipeline, ohe_cols)
    ])

    # Fit-transform train, transform test
    X_train_processed = pd.DataFrame(
        preprocessor.fit_transform(X_train),
        columns=get_feature_names(preprocessor, X_train),
        index=X_train.index
    )
    X_test_processed = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=X_train_processed.columns,
        index=X_test.index
    )

    return X_train_processed, X_test_processed, preprocessor


def get_feature_names(preprocessor, X):
    """
    Extract feature names from a ColumnTransformer (with OneHotEncoder + OrdinalEncoder).
    Works with sklearn >= 1.0
    """
    output_features = []

    for name, transformer, cols in preprocessor.transformers_:
        if transformer == "drop" or transformer == "passthrough":
            # passthrough keeps original names
            if transformer == "passthrough":
                output_features.extend(cols)
            continue

        if isinstance(transformer, Pipeline):
            encoder = transformer.named_steps.get("encoder", None)
        else:
            encoder = transformer

        if hasattr(encoder, "get_feature_names_out"):
            # for OneHotEncoder
            feature_names = encoder.get_feature_names_out(cols)
        else:
            # for OrdinalEncoder or numeric imputer
            feature_names = cols
        output_features.extend(feature_names)

    return output_features


def format_classification_report(report_dict):
    """Convert classification report into a DataFrame for nicer display."""
    report_df = pd.DataFrame(report_dict).transpose()
    return report_df.round(3)


