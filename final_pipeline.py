"""QM640 Data Analytics Capstone - Final Report Pipeline.

The State of Cloud-Native Transformation on Telecommunications Networks in
Brazil: A Municipal-Level Machine Learning Analysis of 5G NR/SA Rollout,
Digital Readiness, and Private-Network Adoption.

Author: Rony Anderson Spada Pedroso (Walsh College, QM640)

Running ``python final_pipeline.py`` regenerates every number, table, and
figure reported in the final report from the committed analysis input
``data/processed/merged_municipal_dataset.csv`` with a fixed seed (42).
Outputs are written to ``reports/figures`` and ``reports/tables``.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
DATA_PATH = os.path.join("data", "processed", "merged_municipal_dataset.csv")
FIG_DIR = os.path.join("reports", "figures")
TAB_DIR = os.path.join("reports", "tables")

REGION_MAP = {
    "1": "North",
    "2": "Northeast",
    "3": "Southeast",
    "4": "South",
    "5": "Center-West",
}

RENAME_MAP = {
    "V614_densidade demografic": "POP_DENSITY",
    "V6318_area da unidade terr": "AREA_KM2",
    "V93_populacao residente": "POP_CENSUS_2022",
    "ERB_NR": "TECH_5G_SA_CNT",
}

EXOG_NUM = ["GDP_PER_CAP", "POP_2024", "POP_DENSITY", "FIBER_PER_100", "LTE_ACCESS_PER_100"]
LEAKY = ["NR_ACCESS_PER_100", "AVG_DL_SPEED"]
CLUSTER_FEATURES = [
    "GDP_PER_CAP",
    "POP_DENSITY",
    "FIBER_PER_100",
    "LTE_ACCESS_PER_100",
    "NR_ACCESS_PER_100",
    "AVG_DL_SPEED",
]

RESULTS: dict = {}


def load_and_clean() -> pd.DataFrame:
    """Load the supplied merged dataset and restore one row per municipality."""
    raw = pd.read_csv(DATA_PATH, dtype={"MUNICIP_ID": str})
    RESULTS["raw_shape"] = list(raw.shape)
    RESULTS["raw_unique_municipalities"] = int(raw["MUNICIP_ID"].nunique())

    # 1) Drop exact duplicate rows.
    exact_dupes = int(raw.duplicated().sum())
    df = raw.drop_duplicates().copy()
    RESULTS["exact_duplicate_rows_dropped"] = exact_dupes
    RESULTS["rows_after_exact_dedup"] = int(len(df))

    # 2) Collapse duplicate municipality keys.
    dup_key_rows = int(len(df) - df["MUNICIP_ID"].nunique())
    RESULTS["duplicate_key_rows"] = dup_key_rows
    agg = {col: "first" for col in df.columns if col != "MUNICIP_ID"}
    agg["SLP_STATION_CNT"] = "sum"
    agg["PRIVATE_5G_LIC"] = "max"

    def first_non_null(series: pd.Series):
        non_null = series.dropna()
        return non_null.iloc[0] if len(non_null) else np.nan

    for col, how in list(agg.items()):
        if how == "first":
            agg[col] = first_non_null
    df = df.groupby("MUNICIP_ID", as_index=False).agg(agg)
    RESULTS["clean_rows"] = int(len(df))

    # 3) Align variable names with the project data dictionary.
    df = df.rename(columns=RENAME_MAP)

    # 4) Feature engineering: region, per-capita/per-area intensities, log1p.
    df["REGION"] = df["MUNICIP_ID"].str[0].map(REGION_MAP)
    df["FIBER_PER_100"] = 100.0 * df["FIBER_ACCESSES"] / df["POP_2024"]
    df["LTE_ACCESS_PER_100"] = 100.0 * df["ACC_LTE"] / df["POP_2024"]
    df["NR_ACCESS_PER_100"] = 100.0 * df["ACC_NR"] / df["POP_2024"]
    df["SA_ERB_PER_100K_POP"] = 100_000.0 * df["TECH_5G_SA_CNT"] / df["POP_2024"]
    df["SA_ERB_PER_1000_KM2"] = 1_000.0 * df["TECH_5G_SA_CNT"] / df["AREA_KM2"]
    df["SLP_PER_100K_POP"] = 100_000.0 * df["SLP_STATION_CNT"] / df["POP_2024"]
    for col in [
        "POP_2024",
        "GDP_PER_CAP",
        "POP_DENSITY",
        "FIBER_PER_100",
        "LTE_ACCESS_PER_100",
        "NR_ACCESS_PER_100",
        "TECH_5G_SA_CNT",
        "SA_ERB_PER_100K_POP",
        "SLP_PER_100K_POP",
        "AVG_DL_SPEED",
    ]:
        df[f"LOG1P_{col}"] = np.log1p(df[col])
    RESULTS["clean_columns"] = int(df.shape[1])

    # 5) Targets.
    sa = df["TECH_5G_SA_CNT"]
    RESULTS["sa_p70"] = float(sa.quantile(0.70))
    RESULTS["sa_p75"] = float(sa.quantile(0.75))
    RESULTS["sa_p80"] = float(sa.quantile(0.80))
    df["HIGH_SA_P70"] = (sa > sa.quantile(0.70)).astype(int)
    df["HIGH_SA_P75"] = (sa > sa.quantile(0.75)).astype(int)
    df["HIGH_SA_P80"] = (sa > sa.quantile(0.80)).astype(int)
    slp_p75 = df["SLP_PER_100K_POP"].quantile(0.75)
    RESULTS["slp_p75"] = float(slp_p75)
    df["HIGH_SLP"] = (df["SLP_PER_100K_POP"] > slp_p75).astype(int)

    RESULTS["share_with_nr"] = float((sa > 0).mean())
    RESULTS["private_5g_lic_mean"] = float(df["PRIVATE_5G_LIC"].mean())
    return df


def descriptives(df: pd.DataFrame) -> None:
    """Missingness profile and descriptive statistics (Tables 6-7, Figure 1)."""
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    miss_tab = pd.DataFrame(
        {
            "missing_count": miss,
            "missing_pct": (100.0 * miss / len(df)).round(4),
            "dtype": [str(df[c].dtype) for c in miss.index],
        }
    )
    miss_tab.to_csv(os.path.join(TAB_DIR, "table07_missingness.csv"))

    desc_vars = [
        "POP_2024",
        "POP_DENSITY",
        "GDP_PER_CAP",
        "FIBER_PER_100",
        "LTE_ACCESS_PER_100",
        "NR_ACCESS_PER_100",
        "TECH_5G_SA_CNT",
        "SA_ERB_PER_100K_POP",
        "AVG_DL_SPEED",
        "SLP_PER_100K_POP",
    ]
    desc = df[desc_vars].agg(["mean", "std", "median", lambda s: s.quantile(0.75), "max"]).T
    desc.columns = ["mean", "sd", "median", "p75", "max"]
    desc.round(1).to_csv(os.path.join(TAB_DIR, "table06_descriptives.csv"))
    RESULTS["complete_cases"] = int(df[EXOG_NUM + ["REGION"]].dropna().shape[0])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    miss_pct = (100.0 * miss / len(df)).sort_values()
    ax.barh(miss_pct.index, miss_pct.values, color="#1f77b4")
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Missingness profile after cleaning")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figure01_missingness.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, col in zip(
        axes.ravel(),
        ["GDP_PER_CAP", "POP_DENSITY", "FIBER_PER_100", "TECH_5G_SA_CNT", "AVG_DL_SPEED", "SA_ERB_PER_100K_POP"],
    ):
        ax.hist(df[col].dropna(), bins=50, color="#1f77b4")
        ax.set_title(col, fontsize=9)
        ax.set_ylabel("Municipalities", fontsize=8)
    fig.suptitle("Distribution diagnostics for key variables")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figure02_distributions.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5))
    for j, col in enumerate(["POP_2024", "GDP_PER_CAP", "TECH_5G_SA_CNT"]):
        axes[0, j].hist(df[col].dropna(), bins=50, color="#1f77b4")
        axes[0, j].set_title(f"{col} (raw)", fontsize=9)
        axes[1, j].hist(np.log1p(df[col].dropna()), bins=50, color="#ff7f0e")
        axes[1, j].set_title(f"{col} (log1p)", fontsize=9)
    fig.suptitle("Skewed variables on raw versus log1p scales")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figure03_log1p.png"), dpi=200)
    plt.close(fig)

    corr_vars = [
        "GDP_PER_CAP",
        "POP_DENSITY",
        "POP_2024",
        "FIBER_PER_100",
        "LTE_ACCESS_PER_100",
        "NR_ACCESS_PER_100",
        "TECH_5G_SA_CNT",
        "SA_ERB_PER_100K_POP",
        "AVG_DL_SPEED",
        "SLP_PER_100K_POP",
    ]
    corr = df[corr_vars].corr()
    corr.round(2).to_csv(os.path.join(TAB_DIR, "correlation_matrix.csv"))
    RESULTS["corr_speed_nr"] = float(corr.loc["AVG_DL_SPEED", "NR_ACCESS_PER_100"])
    RESULTS["corr_speed_density"] = float(corr.loc["AVG_DL_SPEED", "POP_DENSITY"])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(corr_vars)))
    ax.set_xticklabels(corr_vars, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(corr_vars)))
    ax.set_yticklabels(corr_vars, fontsize=7)
    for i in range(len(corr_vars)):
        for j in range(len(corr_vars)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6, color="black")
    fig.colorbar(im)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figure04_correlation.png"), dpi=200)
    plt.close(fig)


def rollout_stage_analysis(df: pd.DataFrame) -> None:
    """Download speed by rollout stage (Table 8, Figure 5) and regional EDA."""
    import matplotlib.pyplot as plt

    p75 = df["TECH_5G_SA_CNT"].quantile(0.75)
    stage = np.where(
        df["TECH_5G_SA_CNT"] > p75,
        "High-density NR/SA (> P75)",
        np.where(df["TECH_5G_SA_CNT"] > 0, "NR present below P75", "No NR station"),
    )
    df = df.assign(ROLLOUT_STAGE=stage)
    tab = (
        df.groupby("ROLLOUT_STAGE")
        .agg(
            n=("MUNICIP_ID", "count"),
            mean_speed=("AVG_DL_SPEED", "mean"),
            median_speed=("AVG_DL_SPEED", "median"),
            sd_speed=("AVG_DL_SPEED", "std"),
            median_fiber=("FIBER_PER_100", "median"),
            median_sa=("SA_ERB_PER_100K_POP", "median"),
        )
        .round(2)
    )
    tab.to_csv(os.path.join(TAB_DIR, "table08_speed_by_stage.csv"))
    RESULTS["stage_counts"] = tab["n"].to_dict()
    RESULTS["stage_table"] = tab.round(2).to_dict()

    groups = [g["AVG_DL_SPEED"].dropna().values for _, g in df.groupby("ROLLOUT_STAGE")]
    f_stat, p_val = stats.f_oneway(*groups)
    RESULTS["anova_f"] = float(f_stat)
    RESULTS["anova_p"] = float(p_val)

    order = ["No NR station", "NR present below P75", "High-density NR/SA (> P75)"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.boxplot(
        [df.loc[df["ROLLOUT_STAGE"] == s, "AVG_DL_SPEED"].dropna() for s in order],
        showfliers=False,
    )
    # Set tick labels explicitly: works on every matplotlib version
    # (the boxplot keyword was renamed labels -> tick_labels in 3.9).
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order)
    ax.set_ylabel("Average download speed (Mbps)")
    ax.set_title("Download speed by NR/SA rollout stage")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figure05_speed_by_stage.png"), dpi=200)
    plt.close(fig)

    reg = (
        df.assign(with_nr=(df["TECH_5G_SA_CNT"] > 0).astype(float))
        .groupby("REGION")["with_nr"]
        .agg(["count", "mean"])
    )
    reg["share_pct"] = (100 * reg["mean"]).round(1)
    reg.to_csv(os.path.join(TAB_DIR, "regional_nr_presence.csv"))
    RESULTS["regional_nr_share"] = reg["share_pct"].to_dict()

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    reg_sorted = reg.sort_values("share_pct")
    ax.bar(reg_sorted.index, reg_sorted["share_pct"], color="#1f77b4")
    for i, v in enumerate(reg_sorted["share_pct"]):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Municipalities with any NR/SA deployment (%)")
    ax.set_title("Share of municipalities with any 5G NR/SA deployment by region")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figure06_region_share.png"), dpi=200)
    plt.close(fig)

    sub = df[df["TECH_5G_SA_CNT"] > 0]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(sub["FIBER_PER_100"], sub["SA_ERB_PER_100K_POP"], s=6, alpha=0.4)
    ax.set_yscale("log")
    ax.set_xlabel("Fiber accesses per 100 inhabitants")
    ax.set_ylabel("NR/SA ERBs per 100,000 population (log scale)")
    ax.set_title("Fiber intensity versus NR/SA station density (NR-present municipalities)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figure07_fiber_vs_sa.png"), dpi=200)
    plt.close(fig)


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = (
        ["MUNICIP_ID", "REGION"]
        + EXOG_NUM
        + LEAKY
        + [
            "TECH_5G_SA_CNT",
            "SA_ERB_PER_100K_POP",
            "SLP_PER_100K_POP",
            "HIGH_SA_P70",
            "HIGH_SA_P75",
            "HIGH_SA_P80",
            "HIGH_SLP",
        ]
    )
    frame = df[cols].dropna().reset_index(drop=True)
    frame = pd.get_dummies(frame, columns=["REGION"], drop_first=True)
    RESULTS["model_n"] = int(len(frame))
    return frame


def rq1_classification(frame: pd.DataFrame) -> None:
    region_dummies = [c for c in frame.columns if c.startswith("REGION_")]
    exog = EXOG_NUM + region_dummies

    def run_rf(features: list, target: str, tune: bool = False) -> dict:
        x = frame[features].astype(float)
        y = frame[target]
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=0.25, stratify=y, random_state=SEED
        )
        if tune:
            grid = GridSearchCV(
                RandomForestClassifier(random_state=SEED, class_weight="balanced"),
                {
                    "n_estimators": [300, 600],
                    "max_depth": [None, 8, 14],
                    "min_samples_leaf": [1, 5],
                },
                scoring="f1",
                cv=5,
                n_jobs=-1,
            )
            grid.fit(x_tr, y_tr)
            model = grid.best_estimator_
            cv_scores = cross_val_score(model, x_tr, y_tr, scoring="f1", cv=5, n_jobs=-1)
            best = grid.best_params_
        else:
            model = RandomForestClassifier(
                n_estimators=300, random_state=SEED, class_weight="balanced"
            )
            model.fit(x_tr, y_tr)
            cv_scores = cross_val_score(model, x_tr, y_tr, scoring="f1", cv=5, n_jobs=-1)
            best = None
        pred = model.predict(x_te)
        proba = model.predict_proba(x_te)[:, 1]
        return {
            "accuracy": float(accuracy_score(y_te, pred)),
            "f1": float(f1_score(y_te, pred)),
            "roc_auc": float(roc_auc_score(y_te, proba)),
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_sd": float(cv_scores.std()),
            "positive_share": float(y.mean()),
            "best_params": best,
            "importances": dict(
                zip(features, [float(v) for v in model.feature_importances_])
            ),
        }

    RESULTS["rq1_leakage_free_p75"] = run_rf(exog, "HIGH_SA_P75", tune=True)
    RESULTS["rq1_naive_p75"] = run_rf(exog + LEAKY, "HIGH_SA_P75", tune=False)
    RESULTS["rq1_leakage_free_p70"] = run_rf(exog, "HIGH_SA_P70", tune=False)
    RESULTS["rq1_leakage_free_p80"] = run_rf(exog, "HIGH_SA_P80", tune=False)

    import matplotlib.pyplot as plt

    imp = pd.Series(RESULTS["rq1_leakage_free_p75"]["importances"]).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(imp.index, imp.values, color="#1f77b4")
    ax.set_xlabel("Random forest feature importance")
    ax.set_title("RQ1 leakage-free feature importances (P75 target)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figure08_rq1_importances.png"), dpi=200)
    plt.close(fig)


def rq2_ridge(frame: pd.DataFrame) -> None:
    region_dummies = [c for c in frame.columns if c.startswith("REGION_")]
    exog = EXOG_NUM + region_dummies
    alphas = np.logspace(-2, 3, 30)

    def run_ridge(features: list, target_values: pd.Series, sub: pd.DataFrame) -> dict:
        x = sub[features].astype(float)
        y = target_values
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, random_state=SEED)
        pipe = Pipeline(
            [("scale", StandardScaler()), ("ridge", RidgeCV(alphas=alphas, cv=5))]
        )
        pipe.fit(x_tr, y_tr)
        pred = pipe.predict(x_te)
        return {
            "r2": float(r2_score(y_te, pred)),
            "mae": float(mean_absolute_error(y_te, pred)),
            "alpha": float(pipe.named_steps["ridge"].alpha_),
            "coefficients": dict(
                zip(features, [float(v) for v in pipe.named_steps["ridge"].coef_])
            ),
        }

    RESULTS["rq2_raw"] = run_ridge(exog, frame["SA_ERB_PER_100K_POP"], frame)
    RESULTS["rq2_log"] = run_ridge(exog, np.log1p(frame["SA_ERB_PER_100K_POP"]), frame)
    RESULTS["rq2_naive_raw"] = run_ridge(exog + LEAKY, frame["SA_ERB_PER_100K_POP"], frame)
    nr_present = frame[frame["TECH_5G_SA_CNT"] > 0]
    RESULTS["rq2_nr_present_n"] = int(len(nr_present))
    RESULTS["rq2_conditional_log"] = run_ridge(
        exog, np.log1p(nr_present["SA_ERB_PER_100K_POP"]), nr_present
    )


def rq3_clustering(df: pd.DataFrame) -> None:
    sub = df[CLUSTER_FEATURES + ["REGION", "MUNICIP_ID"]].dropna().reset_index(drop=True)
    x = StandardScaler().fit_transform(np.log1p(sub[CLUSTER_FEATURES]))

    sils = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(x)
        sils[k] = float(silhouette_score(x, km.labels_))
    RESULTS["rq3_silhouettes"] = sils
    best_k = max(sils, key=sils.get)
    RESULTS["rq3_best_k"] = int(best_k)

    km = KMeans(n_clusters=best_k, random_state=SEED, n_init=10).fit(x)
    labels = km.labels_
    profile = sub[CLUSTER_FEATURES].groupby(labels).mean().round(1)
    # Identify the higher-readiness cluster by GDP per capita.
    hi = int(profile["GDP_PER_CAP"].idxmax())
    RESULTS["rq3_cluster_sizes"] = pd.Series(labels).value_counts().to_dict()
    RESULTS["rq3_higher_cluster"] = hi
    RESULTS["rq3_profiles"] = profile.to_dict()
    profile.to_csv(os.path.join(TAB_DIR, "rq3_cluster_profiles.csv"))

    ct = pd.crosstab(sub["REGION"], labels)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    RESULTS["rq3_chi2"] = float(chi2)
    RESULTS["rq3_chi2_p"] = float(p)
    RESULTS["rq3_chi2_dof"] = int(dof)
    share_hi = (ct[hi] / ct.sum(axis=1) * 100).round(1)
    RESULTS["rq3_share_higher_by_region"] = share_hi.to_dict()
    ct.to_csv(os.path.join(TAB_DIR, "rq3_region_crosstab.csv"))

    rng = np.random.RandomState(SEED)
    aris = []
    for _ in range(20):
        idx = rng.choice(len(x), size=len(x), replace=True)
        km_b = KMeans(n_clusters=best_k, random_state=SEED, n_init=10).fit(x[idx])
        aris.append(adjusted_rand_score(labels[idx], km_b.labels_))
    RESULTS["rq3_ari_mean"] = float(np.mean(aris))
    RESULTS["rq3_ari_sd"] = float(np.std(aris))


def rq4_logistic(frame: pd.DataFrame) -> None:
    import statsmodels.api as sm

    region_dummies = [c for c in frame.columns if c.startswith("REGION_")]
    exog = EXOG_NUM + region_dummies
    x = frame[exog].astype(float)
    y = frame["HIGH_SLP"]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, stratify=y, random_state=SEED)

    scaler = StandardScaler().fit(x_tr)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    clf.fit(scaler.transform(x_tr), y_tr)
    pred = clf.predict(scaler.transform(x_te))
    proba = clf.predict_proba(scaler.transform(x_te))[:, 1]
    RESULTS["rq4"] = {
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "recall": float(recall_score(y_te, pred)),
        "accuracy": float(accuracy_score(y_te, pred)),
        "positive_share": float(y.mean()),
        "coefficients": dict(zip(exog, [float(v) for v in clf.coef_[0]])),
    }

    x_sm = sm.add_constant(StandardScaler().fit_transform(x))
    fit = sm.Logit(y, x_sm).fit(disp=0)
    RESULTS["rq4"]["llr_p"] = float(fit.llr_pvalue)


def workflow_diagram():
    """Figure 1 of the final report: end-to-end workflow of the pipeline."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

    def box(x, y, w, h, title, lines, fc="#e8f0fe", ec="#1a56a0"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6", fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=9.5, weight="bold", color="#123c6b")
        ax.text(x + w / 2, y + h - 6.2, "\n".join(lines), ha="center", va="top", fontsize=7.6, color="#222222")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#444444"))

    green = dict(fc="#e9f7ec", ec="#1f7a33")
    box(1, 44, 28, 14, "Data sources (open government)",
        ["Anatel: mobile accesses, licensed", "stations (ERB), SLP registry, fiber,", "download speed",
         "IBGE SIDRA: population, GDP,", "area, density"], fc="#fdf1e3", ec="#b06a12")
    box(36, 44, 28, 14, "Acquisition & merge",
        ["Wayback snapshots + data-panel", "extraction + file-server metadata", "Join key: 7-digit IBGE code",
         "merged_municipal_dataset.csv", "(10,107 x 25; 5,571 municipalities)"])
    box(71, 44, 28, 14, "Cleaning (final_pipeline.py)",
        ["Drop 193 exact duplicates", "Collapse 4,343 duplicate keys", "Rename to data dictionary",
         "One row per municipality", "(5,571 x 42)"])
    arrow(29, 51, 36, 51); arrow(64, 51, 71, 51)
    box(71, 24, 28, 14, "Feature engineering",
        ["Per-capita intensities (FIBER_PER_100,", "LTE/NR_ACCESS_PER_100,", "SA_ERB_PER_100K_POP, SLP_PER_100K)",
         "log1p transforms; REGION from code", "Targets: HIGH_SA_P75, HIGH_SLP"])
    box(36, 24, 28, 14, "EDA",
        ["Missingness profile (max 0.11%)", "Skew & zero-inflation diagnostics", "Correlation structure",
         "Speed by rollout stage (ANOVA)", "Regional deployment shares"])
    box(1, 24, 28, 14, "Leakage audit",
        ["Exclude NR_ACCESS_PER_100 &", "AVG_DL_SPEED from RQ1/RQ2", "Exogenous set: GDP, population,",
         "density, fiber & LTE intensity,", "region dummies"], fc="#fdeaea", ec="#a03a3a")
    arrow(85, 44, 85, 38); arrow(71, 31, 64, 31); arrow(36, 31, 29, 31)
    box(1, 3, 22.5, 15, "RQ1 Random forest",
        ["High-density NR/SA class", "5-fold CV grid tuning", "Stratified 75/25, seed 42",
         "Acc/F1/ROC-AUC +", "P70/P75/P80 sensitivity"], **green)
    box(26.5, 3, 22.5, 15, "RQ2 Ridge regression",
        ["SA_ERB_PER_100K_POP", "RidgeCV (5-fold, alpha grid)", "Raw, log1p, conditional",
         "R-squared, MAE,", "standardized coefficients"], **green)
    box(52, 3, 22.5, 15, "RQ3 K-means + chi-square",
        ["k = 2-6 by silhouette", "20-resample bootstrap ARI", "Cluster profiles",
         "Chi-square vs. region", "(alpha = .05)"], **green)
    box(77.5, 3, 21.5, 15, "RQ4 Logistic regression",
        ["HIGH_SLP intensity target", "Balanced class weights", "statsmodels LLR test",
         "ROC-AUC, recall,", "coefficients"], **green)
    arrow(12, 24, 12, 18); arrow(15, 24, 37, 18); arrow(15, 24, 62, 18); arrow(15, 24, 87, 18)
    ax.text(50, 0.6, "Outputs: reports/figures, reports/tables, headline_results.json  "
            "(fixed seed = 42; single command: python final_pipeline.py)",
            ha="center", fontsize=8.2, style="italic", color="#333333")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figure00_workflow.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    np.random.seed(SEED)

    workflow_diagram()
    df = load_and_clean()
    descriptives(df)
    rollout_stage_analysis(df)
    frame = model_frame(df)
    rq1_classification(frame)
    rq2_ridge(frame)
    rq3_clustering(df)
    rq4_logistic(frame)

    with open(os.path.join(TAB_DIR, "headline_results.json"), "w") as fh:
        json.dump(RESULTS, fh, indent=2, default=str)
    print(json.dumps(RESULTS, indent=2, default=str))


if __name__ == "__main__":
    main()
