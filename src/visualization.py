import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

import plotly.figure_factory as ff


def plot_target_distribution(df, target_col="loan_status"):
    fig = px.histogram(df, x=target_col, color=target_col,
                       color_discrete_sequence=["skyblue","salmon"],
                       title="Target Distribution (Loan Default)")
    return fig


def plot_missing_values(df):
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        return None
    else:
        fig = px.bar(x=missing.index, y=missing.values,
                     labels={'x':'Column', 'y':'Missing Values Count'},
                     title="Missing Values Count per Column",
                     color=missing.values, color_continuous_scale='viridis')
        return fig


def plot_numerical_distributions(df, num_cols):
    figs = []
    for col in num_cols:
        fig = px.histogram(df, x=col, nbins=30, marginal="box",
                           color_discrete_sequence=["steelblue"],
                           title=f"Distribution of {col}")
        figs.append(fig)
    return figs


def plot_numerical_vs_target(df, num_cols, target_col="loan_status"):
    figs = []
    for col in num_cols:
        fig = px.box(df, x=target_col, y=col, color=target_col,
                     color_discrete_sequence=["skyblue","salmon"],
                     title=f"{col} vs {target_col}")
        figs.append(fig)
    return figs


def plot_categorical_distributions(df, cat_cols, target_col="loan_status"):
    figs = []
    for col in cat_cols:
        fig = px.histogram(df, x=col, color=target_col, barmode='group',
                           color_discrete_sequence=["skyblue","salmon"],
                           title=f"{col} by {target_col}")
        figs.append(fig)
    return figs


def plot_correlation_heatmap(df, num_cols):
    corr = df[num_cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation")
    ))
    fig.update_layout(title="Correlation Heatmap", width=800, height=700)
    return fig


def plot_default_rate_by_categorical(df, cat_cols, target_col="loan_status"):
    figs = []
    for col in cat_cols:
        default_rate = df.groupby(col)[target_col].mean().sort_values()
        fig = px.bar(x=default_rate.index, y=default_rate.values,
                     labels={'x':col, 'y':'Proportion of Defaults'},
                     title=f"Default Rate by {col}",
                     color=default_rate.values, color_continuous_scale='rocket')
        figs.append(fig)
    return figs


def plot_confusion_matrix(y_true, y_pred):
    """Return a Plotly confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    z = cm.tolist()
    x = [f"Predicted {i}" for i in range(len(cm))]
    y = [f"Actual {i}" for i in range(len(cm))]

    fig = ff.create_annotated_heatmap(
        z=z,
        x=x,
        y=y,
        colorscale="Blues",
        showscale=True,
        annotation_text=[[str(val) for val in row] for row in z],
        font_colors=["black"],   # make text black
    )
    for ann in fig.layout.annotations:
        ann.font.size = 27  # increase annotation text size

    fig.update_layout(

        xaxis=dict(title="Predicted"),
        yaxis=dict(title="Actual"),
        yaxis_autorange="reversed",
        width=300,   # 🔹 shrink overall width
        height=250,  # 🔹 shrink overall height
        margin=dict(l=40, r=40, t=10, b=10)  # adjust margins
    )

    return fig


def plot_roc_curves(y_true, y_probas: dict):
    """
    Plot ROC curves for multiple models using Plotly.

    Parameters:
    - y_true: array-like, true labels
    - y_probas: dict, model_name -> predicted probabilities (for positive class)
    """
    fig = go.Figure()

    for model_name, y_proba in y_probas.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{model_name} (AUC={roc_auc:.3f})"
        ))

    # Random guess line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Guess",
        line=dict(color="black", dash="dash")
    ))

    # Layout
    fig.update_layout(
        xaxis=dict(title="False Positive Rate"),
        yaxis=dict(title="True Positive Rate"),
        width=400,
        height=350,
        legend=dict(x=0.02, y=0.02, bordercolor="black", borderwidth=1),
        margin = dict(l=20, r=20, t=5, b=0)
    )

    return fig



def plot_pr_curves(y_true, y_probas: dict):
    """
    Plot Precision-Recall curves for multiple models using Plotly.

    Parameters:
    - y_true: array-like, true labels
    - y_probas: dict, model_name -> predicted probabilities (for positive class)
    """
    fig = go.Figure()

    for model_name, y_proba in y_probas.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode="lines",
            name=f"{model_name} (AP={avg_precision:.3f})"
        ))

    # Baseline (proportion of positives in dataset)
    baseline = sum(y_true) / len(y_true)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[baseline, baseline],
        mode="lines",
        name="Baseline",
        line=dict(color="black", dash="dash")
    ))

    # Layout
    fig.update_layout(
        xaxis=dict(title="Recall"),
        yaxis=dict(title="Precision"),
        width=400,
        height=350,
        legend=dict(x=0.02, y=0.02, bordercolor="black", borderwidth=1),
        margin=dict(l=20, r=20, t=5, b=0)
    )

    return fig
