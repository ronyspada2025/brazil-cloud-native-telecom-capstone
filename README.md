# QM640 Data Analytics Capstone

## The State of Cloud-Native Transformation on Telecommunications Networks in Brazil

**Author:** Rony Anderson Spada Pedroso
**Institution:** Walsh College
**Course:** QM640: Data Analytics Capstone
**Mentor:** Sridhar Srinivas
**Current report version:** Final report (four-RQ, leakage-audited design)

## Project Overview

This repository supports a data analytics capstone project evaluating the municipal-level state of cloud-native telecommunications transformation in Brazil. It uses open public data from the Brazilian National Telecommunications Agency (Anatel) and the Brazilian Institute of Geography and Statistics (IBGE) to analyze where 5G NR/SA rollout, fiber intensity, digital readiness, and private-network/SLP adoption are concentrated across all 5,571 municipalities.

Two questions from the original synopsis — a network-generation QoS comparison and a customer-satisfaction prediction — were removed at the interim stage because the merged municipal dataset does not include `DROP_RATE` or `SATISFACTION`; they remain documented future extensions.

## Abstract

**Problem.** Brazil's migration toward cloud-native, 5G Standalone (SA) network architecture is highly unequal across its 5,571 municipalities, and it is poorly understood which municipal characteristics genuinely distinguish where that infrastructure lands, leaving carriers and the regulator without an evidence base for allocating capital and monitoring the coverage obligations attached to the 2021 spectrum auction.

**Solution approach.** Classical inferential statistics (one-way ANOVA, chi-square test of independence, likelihood-ratio test) combined with machine learning (cross-validated random forest, ridge regression, k-means with bootstrap stability analysis, balanced logistic regression) under a leakage-free specification protocol that restricts predictors to structurally exogenous variables; all supervised models evaluated on stratified 75/25 held-out partitions with 5-fold cross-validated tuning and a fixed seed (42).

**Data.** Open administrative records from Anatel (mobile accesses, licensed stations, private-network/SLP registry, fiber accesses, measured download speeds) merged with IBGE socioeconomic aggregates at the municipal grain; 10,107 supplied records cleaned to the full national frame of N = 5,571 municipalities.

**Major results.** RQ1 random forest, leakage-free: accuracy = .915, F1 = .808, ROC-AUC = .927 (naive specification .965). RQ2 ridge: R² = .02 (raw) to .11 (NR-present municipalities). RQ3: two readiness clusters, bootstrap ARI = .99, associated with macro-region, χ²(4) = 254.1, p < .001. RQ4 logistic: ROC-AUC = .810, recall = .753; GDP per capita and population are the strongest drivers.

**Implementation area.** Reproducible municipal scores (rollout probability, readiness cluster, SLP-intensity probability) keyed to the IBGE code, usable as a coverage-obligation watchlist by Anatel and as an investment-prioritization signal by Claro, Vivo, and TIM.

## Research Questions

- **RQ1 — Rollout classification.** Can exogenous socioeconomic and infrastructure characteristics predict whether a municipality is a high-density 5G NR/SA rollout site?
- **RQ2 — Infrastructure drivers of station density.** Which exogenous characteristics explain the population-normalized intensity of NR/SA station deployment?
- **RQ3 — Digital-readiness segmentation.** How do municipalities cluster on digital-readiness indicators, and is cluster membership independent of geographic region?
- **RQ4 — Private-network/SLP intensity drivers.** Which municipal characteristics drive high private-network (SLP) deployment intensity?

## Headline Results (held-out, seed = 42)

| RQ | Model | Key metrics |
|---|---|---|
| RQ1 | Random forest (leakage-free, tuned) | Accuracy .915, F1 .808, ROC-AUC .927 (naive spec: .965) |
| RQ2 | RidgeCV | R² .02 (raw), .06 (log1p), .11 (NR-present conditional) |
| RQ3 | K-means (k = 2) + chi-square | Silhouette .305, bootstrap ARI .99, χ²(4) = 254.1, p < .001 |
| RQ4 | Balanced logistic regression | ROC-AUC .810, recall .753, LLR p < .001 |

## Reproduction

The authoritative reproduction path is the single-command pipeline:

```bash
pip install -r requirements.txt
python final_pipeline.py
```

`requirements.txt` pins `scikit-learn==1.8.0`, the version used for the reported random-forest results (tree construction changed in 1.9; every other result is version-stable). Use `python3` on macOS/Linux if `python` is not on your PATH.

**Cross-platform note.** On Linux x86-64 (the environment used for the final report, and the one Google Colab provides) every value regenerates exactly. On Apple Silicon macOS (Python 3.13, scikit-learn 1.8.0) all cleaning, regression, clustering, chi-square, and logistic results are identical, while random-forest metrics shift by at most ±.001 (e.g., F1 .809 vs. .808; naive ROC-AUC .966 vs. .965) because floating-point summation order differs across processor architectures; no conclusion changes. The committed `reports/` outputs were generated on Apple Silicon.

It regenerates every number, table, and figure of the final report from the committed analysis input (`data/processed/merged_municipal_dataset.csv`) with a fixed seed (42), writing outputs to `reports/figures` and `reports/tables` (including a machine-readable `headline_results.json`).

`notebooks/06_final_report_pipeline_colab.ipynb` runs the identical pipeline in Google Colab (~3 minutes), prints a headline-results check against the final report, and packages all outputs for download.

## Repository Structure

```text
.
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── final_pipeline.py            (authoritative reproduction path)
├── data/
│   ├── raw/                     (git-ignored; downloaded locally per src/data_loader.py)
│   └── processed/
│       └── merged_municipal_dataset.csv   (committed analysis input, 1.8 MB)
├── docs/
│   └── DATA_DICTIONARY.md
├── notebooks/
│   ├── 01_data_cleaning_and_alignment.ipynb   (scaffold: original acquisition design)
│   ├── 02_feature_engineering.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_statistical_tests_rq2_rq4.ipynb
│   ├── 05_ml_pipelines_rq1_rq3_rq5_rq6.ipynb
│   └── 06_final_report_pipeline_colab.ipynb   (Colab mirror of final_pipeline.py)
├── reports/
│   ├── figures/                 (Figures 1–9 of the final report)
│   └── tables/                  (result tables + headline_results.json)
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── feature_engineering.py
    ├── model_evaluation.py
    └── visualization_utils.py
```

The five scaffold notebooks document the original seven-extract acquisition design from the synopsis; `final_pipeline.py` and its Colab counterpart are the authoritative reproduction paths for the final report.

## Data Notes

- The supplied merged dataset contains 10,107 rows × 25 columns with exactly 5,571 unique municipalities; `final_pipeline.py` drops 193 exact duplicates, collapses 4,343 duplicate municipality keys (first non-null for stable variables; sum for `SLP_STATION_CNT`; max for `PRIVATE_5G_LIC`), and restores the one-row-per-municipality frame (5,571 × 42 after feature engineering).
- The binary fiber flag is saturated (100% of non-missing values equal 1); `FIBER_PER_100` is the substantive fiber-readiness measure.
- `NR_ACCESS_PER_100` and `AVG_DL_SPEED` are excluded from the RQ1/RQ2 predictor sets as near-target (leakage) variables.
- Raw Anatel/IBGE extracts are excluded from version control because of their size; acquisition is documented in `src/data_loader.py` (Wayback snapshots, data-panel extraction, and file-server metadata parsing). Source landing pages were verified July 23, 2026; raw extracts acquired August 2026.

## License

MIT — see `LICENSE`.
