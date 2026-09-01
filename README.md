# QM640 Data Analytics Capstone

## Municipal Correlates of Licensed 5G NR Infrastructure in Brazil
### A Machine Learning Analysis of Rollout, Digital-Infrastructure Readiness, and Private-Network (SLP) Station Intensity as Proxies for Cloud-Native Transformation

**Author:** Rony Anderson Spada Pedroso
**Institution:** Walsh College
**Course:** QM640: Data Analytics Capstone
**Mentor:** Sridhar Srinivas
**Current report version:** Final report, revised version 1 (September 2026) — responds to the evaluator's assessment: construct-valid naming (NR stations as a proxy; SLP ≠ private 5G), per-capita RQ1 target, two-part count model for RQ2, state-grouped cross-validation, Cramér's V, corrected ROC-AUC reference, associational language throughout

## Project Overview

This repository supports a data analytics capstone project evaluating the municipal-level state of cloud-native telecommunications transformation in Brazil. It uses open public data from the Brazilian National Telecommunications Agency (Anatel) and the Brazilian Institute of Geography and Statistics (IBGE) to analyze where 5G NR rollout, fiber intensity, digital readiness, and private-network/SLP adoption are concentrated across all 5,571 municipalities.

## Abstract

**Problem.** Licensed 5G New Radio (NR) infrastructure is spread unevenly across Brazil's 5,571 municipalities, and it is not sufficiently understood how municipal demographic, economic, and existing telecommunications characteristics are associated with that distribution. NR station counts are used as an observable proxy for cloud-native transformation; they do not establish standalone-core or virtualized operation.

**Solution approach.** Four questions — whether high-count NR rollout can be predicted, how NR station intensity is associated with structural characteristics, how municipalities group on digital-infrastructure readiness, and what is associated with private-network (SLP) station intensity — examined with a random forest, a two-part model (presence logistic plus negative-binomial count model with population exposure), k-means with chi-square and Cramér's V, and logistic regression. Predictors are restricted to pre-existing structural characteristics; validation uses held-out data and state-grouped cross-validation.

**Data.** Open Anatel records merged with IBGE socioeconomic data for all 5,571 municipalities; cross-sectional, non-experimental design.

**Major results.** High-count NR rollout ROC-AUC = .927 (chance .50; prevalence 22.5%), falling to .818 with a per-capita target — much of the signal is municipal scale. NR station intensity is only weakly associated with structure (NB pseudo-R² = .017), with fiber, population, and the South region significant. Two readiness clusters are associated with macro-region (Cramér's V = .21). SLP intensity ROC-AUC = .810, with reduced performance under state-grouped validation (.758 ± .067).

**Implementation area.** A screening and prioritization input for telecommunications planning — not a compliance model — keyed to the IBGE municipal code.

## Research Questions

- **RQ1 — High-count NR rollout classification.** Can pre-existing structural characteristics predict whether a municipality is a high-count licensed 5G NR rollout site, and does the result hold with a per-capita target?
- **RQ2 — Structural correlates of NR station intensity.** To what extent are structural characteristics associated with population-normalized NR station intensity, and which carry significant associations?
- **RQ3 — Digital-infrastructure readiness segmentation.** How do municipalities cluster on readiness indicators, and how strongly is membership associated with region?
- **RQ4 — Correlates of SLP station intensity.** Which characteristics are associated with high private-network-related SLP station intensity, and does the association generalize across states?

## Headline Results (held-out, seed = 42)

| RQ | Model | Key metrics |
|---|---|---|
| RQ1 | Random forest, count target (leakage-free, tuned) | Accuracy .915, F1 .808, ROC-AUC .927 (chance .50; naive spec .965); state-grouped CV AUC .909 ± .023 |
| RQ1 | Random forest, per-capita target (robustness) | Accuracy .813, F1 .590, ROC-AUC .818 |
| RQ2 | Two-part: presence logistic + negative-binomial (population exposure) | Presence AUC .849; NB α = 4.42, pseudo-R² .017, LLR p < .001; ridge R² .02 (raw) – .11 (NR-present) |
| RQ3 | K-means (k = 2) + chi-square | Silhouette .305, bootstrap ARI .99, χ²(4) = 254.1, p < .001, Cramér's V = .21 |
| RQ4 | Balanced logistic regression | ROC-AUC .810, recall .753, LLR p < .001; state-grouped CV AUC .758 ± .067 |

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

## Data Provenance and Reproducibility Scope

Every column of the merged dataset is traced to its IBGE/Anatel source in `docs/DATA_PROVENANCE.md` (Table 3 of the final report), and `reports/figures/figure00b_source_merge.png` shows the merge design. The committed analysis input has SHA-256 `50fac84b16f63d66628741f36686cc076f43c90632a3befe53ca43ed9b316207`; `final_pipeline.py` recomputes and records it in `headline_results.json`, and `python src/data_loader.py --check` verifies the file's columns and checksum.

Reproducibility is exact from the committed merged dataset onward. From the public sources onward it is documented at the dataset level (source, access point, variables, aggregation), not the file level: the specific extract files and download snapshots were not archived at acquisition time. This is stated as a limitation in the final report; re-acquiring and archiving the raw extracts under `data/raw/` is the documented next step. `docs/DATA_PROVENANCE.md` also records each source's reference period or publication cadence (Census 2022; population estimates 2024 vintage; municipal GDP annual with a two-year lag; Anatel accesses monthly; station and SLP registries continuous), so a later rebuild can select the matching vintage where it is known.

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
│   ├── DATA_DICTIONARY.md
│   └── DATA_PROVENANCE.md      (Table 3: source-to-variable lineage)
├── notebooks/
│   ├── 01_data_cleaning_and_alignment.ipynb   (scaffold: original acquisition design)
│   ├── 02_feature_engineering.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_statistical_tests_rq2_rq4.ipynb
│   ├── 05_ml_pipelines_rq1_rq3_rq5_rq6.ipynb
│   └── 06_final_report_pipeline_colab.ipynb   (Colab mirror of final_pipeline.py)
├── reports/
│   ├── figures/                 (Figures 1–10 of the final report)
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

- The supplied merged dataset contains 10,107 rows × 25 columns with exactly 5,571 unique municipalities; `final_pipeline.py` drops 193 exact duplicates, collapses 4,343 duplicate municipality keys (first non-null for stable variables; sum for `SLP_STATION_CNT`; max for `PRIVATE_5G_LIC`), and restores the one-row-per-municipality frame (5,571 × 42 after feature engineering). The duplicate keys have a single origin: 4,343 municipalities appear exactly twice and, within each pair, only `SLP_STATION_CNT` differs — the signature of a one-to-many join with Anatel's SLP registry, which carries more than one record per municipality. Summing `SLP_STATION_CNT` reconstructs the municipal total (e.g., IBGE 1100015: 77 + 31 = 108); keeping only the first record would have discarded 4,343 partial counts.
- The binary fiber flag is saturated (100% of non-missing values equal 1); `FIBER_PER_100` is the substantive fiber-readiness measure.
- `NR_ACCESS_PER_100` and `AVG_DL_SPEED` are excluded from the RQ1/RQ2 predictor sets as near-target (leakage) variables.
- Raw Anatel/IBGE extracts are excluded from version control because of their size; acquisition is documented in `src/data_loader.py` (Wayback snapshots, data-panel extraction, and file-server metadata parsing). Source landing pages were verified July 23, 2026; raw extracts acquired August 2026.

## License

MIT — see `LICENSE`.
