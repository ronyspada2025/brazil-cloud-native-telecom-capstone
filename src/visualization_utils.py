"""Plotting helpers shared by the notebooks (matplotlib/seaborn)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")

REGION_ORDER = ["North", "Northeast", "Center-West", "Southeast", "South"]


def correlation_heatmap(df: pd.DataFrame, cols: list[str],
                        title: str = "Correlation matrix"):
    """Correlation heatmap for the EDA multicollinearity check (notebook 03)."""
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(1.1 * len(cols), 0.9 * len(cols)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def group_boxplot(df: pd.DataFrame, value_col: str, group_col: str,
                  title: str | None = None):
    """Boxplot of a QoS metric across TECH_GENERATION groups (RQ2)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x=group_col, y=value_col, ax=ax)
    ax.set_title(title or f"{value_col} by {group_col}")
    fig.tight_layout()
    return fig


def feature_importance_bar(names: list[str], importances: np.ndarray,
                           title: str = "Feature importance"):
    """Horizontal bar chart of model feature importances (RQ1, RQ5)."""
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(names) + 1.5))
    ax.barh(np.array(names)[order], importances[order])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def cluster_scatter(df: pd.DataFrame, x: str, y: str,
                    cluster_col: str = "cluster",
                    title: str = "K-means clusters"):
    """2-D scatter of municipalities colored by RQ4 cluster."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=cluster_col,
                    palette="deep", s=20, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def roc_curve_plot(y_true, y_score, title: str = "ROC curve"):
    """ROC curve for RQ5's XGBoost classifier."""
    from sklearn.metrics import RocCurveDisplay

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig
