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
| RQ4 | Balanced logistic regression | ROC-AUC .810, recall .747, LLR p < .001 |

## Reproduction

The authoritative reproduction path is the single-command pipeline:

```bash
pip install -r requirements.txt
python final_pipeline.py
```

`requirements.txt` pins `scikit-learn==1.8.0`, the version used for the reported random-forest results (tree construction changed in 1.9; every other result is version-stable). Use `python3` on macOS/Linux if `python` is not on your PATH.

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
