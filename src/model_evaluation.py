"""Evaluation helpers implementing the metrics named in the synopsis.

RQ1: accuracy, F1        RQ2: F statistic, p, Tukey HSD
RQ3: R^2, MAE            RQ4: silhouette coefficient, chi-square
RQ5: ROC-AUC, precision  RQ6: recall, LLR p-value
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics


# ---------- classification (RQ1, RQ5, RQ6) ----------

def classification_report_dict(y_true, y_pred, y_score=None) -> dict:
    """Accuracy, precision, recall, F1 and (if scores given) ROC-AUC."""
    out = {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_true, y_pred, zero_division=0),
        "f1": metrics.f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        out["roc_auc"] = metrics.roc_auc_score(y_true, y_score)
    return out


# ---------- regression (RQ3) ----------

def regression_report_dict(y_true, y_pred) -> dict:
    """R^2 and MAE as defined in the synopsis."""
    return {
        "r2": metrics.r2_score(y_true, y_pred),
        "mae": metrics.mean_absolute_error(y_true, y_pred),
    }


# ---------- ANOVA (RQ2) ----------

@dataclass
class AnovaResult:
    f_statistic: float
    p_value: float


def one_way_anova(groups: list[np.ndarray]) -> AnovaResult:
    """One-way ANOVA across technology-generation groups."""
    f, p = stats.f_oneway(*groups)
    return AnovaResult(f_statistic=float(f), p_value=float(p))


def tukey_hsd(values: pd.Series, labels: pd.Series):
    """Tukey HSD post hoc test (statsmodels). Returns the results object."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    return pairwise_tukeyhsd(endog=values, groups=labels, alpha=0.05)


# ---------- clustering + independence (RQ4) ----------

def silhouette(X, cluster_labels) -> float:
    """Silhouette coefficient for the fitted k-means solution."""
    return float(metrics.silhouette_score(X, cluster_labels))


def chi_square_independence(cluster_labels: pd.Series,
                            regions: pd.Series) -> dict:
    """Chi-square test of independence between clusters and macro-regions."""
    table = pd.crosstab(cluster_labels, regions)
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof),
            "contingency": table}


# ---------- logistic regression significance (RQ6) ----------

def llr_pvalue(fitted_logit) -> float:
    """Log-likelihood-ratio p-value from a fitted statsmodels Logit result."""
    return float(fitted_logit.llr_pvalue)
