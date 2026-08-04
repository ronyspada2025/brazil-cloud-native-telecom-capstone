# QM640 Data Analytics Capstone

## The State of Cloud-Native Transformation on Telecommunications Networks in Brazil

**Author:** Rony Anderson Spada Pedroso  
**Institution:** Walsh College  
**Course:** QM640: Data Analytics Capstone  
**Instructor:** Dr. Srabashi Basu  
**Current report version:** Four-RQ APA 7 interim report with exploratory data analysis  

## Project Overview

This repository supports a data analytics capstone project evaluating the municipal-level state of cloud-native telecommunications transformation in Brazil. The project uses open public data from the Brazilian National Telecommunications Agency (Anatel) and the Brazilian Institute of Geography and Statistics (IBGE) to analyze where 5G NR/SA, fiber intensity, digital-readiness indicators, and private-network/SLP adoption are most concentrated.

The revised interim report uses a four-research-question structure. Two earlier questions from the synopsis—network-generation QoS comparison and customer-satisfaction prediction—were removed from the active interim scope because the uploaded merged municipal dataset does not yet include `DROP_RATE` or `SATISFACTION`. Download speed remains part of the exploratory analysis, but it is no longer treated as a standalone research question.

## Active Research Questions

### RQ1 — 5G NR/SA Rollout Classification

Can municipal socioeconomic markers, population density, fiber intensity, and existing mobile infrastructure predict whether a Brazilian municipality has achieved high-density 5G NR/SA deployment?

### RQ2 — Infrastructure Drivers of 5G NR/SA Station Density

Which socioeconomic and infrastructure characteristics are the strongest predictors of population-normalized 5G NR/SA station density across Brazilian municipalities?

### RQ3 — Municipal Digital-Readiness Segmentation

How do Brazilian municipalities cluster based on digital-transformation readiness indicators, and are these clusters associated with geographic regions?

### RQ4 — Private Network / SLP Adoption Drivers

Which municipal socioeconomic, infrastructure, and mobile-network characteristics influence the likelihood or intensity of private-network/SLP adoption?

## Repository Structure

```text
.
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── merged_municipal_dataset.csv
├── docs/
│   ├── DATA_AVAILABILITY.md
│   ├── DATA_DICTIONARY.md
│   ├── METHODOLOGY.md
│   ├── REPORT_CHANGELOG.md
│   └── RESEARCH_QUESTIONS.md
├── notebooks/
│   ├── 01_data_understanding_and_cleaning.ipynb
│   ├── 02_complete_eda.ipynb
│   └── 03_rq_exploratory_analysis.ipynb
├── reports/
│   ├── interim/
│   │   ├── QM640_Interim_Report_Four_RQs_APA7.docx
│   │   └── QM640_Interim_Report_Four_RQs_APA7.pdf
│   ├── final/
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model_evaluation.py
│   └── visualization_utils.py
└── tests/
```

## Notebook Execution Order

Run the notebooks in this order:

1. `notebooks/01_data_understanding_and_cleaning.ipynb`
2. `notebooks/02_complete_eda.ipynb`
3. `notebooks/03_rq_exploratory_analysis.ipynb`

The first notebook checks the dataset structure, cleans duplicate municipality keys, aligns variable names, and engineers normalized intensity variables. The second notebook performs full exploratory data analysis. The third notebook generates RQ-specific exploratory analyses and preliminary models.

## Interim Data Status

The uploaded merged municipal dataset initially contained 10,107 rows and 25 columns. After duplicate-key correction, the valid municipal analysis frame contains 5,571 rows, matching the intended Brazilian municipality frame.

Key interim findings:

- The municipal frame is complete after cleaning.
- Binary fiber presence is saturated and should not be used as the primary readiness indicator.
- Fiber intensity, measured as fiber accesses per 100 inhabitants, is more useful analytically.
- 5G NR/SA station deployment is highly uneven and zero-inflated.
- Download speed is retained as a supporting EDA variable.
- `DROP_RATE` and `SATISFACTION` are excluded from the active interim scope because they are not present in the uploaded merged dataset.

## Environment Setup

```bash
pip install -r requirements.txt
```

## Reproducibility Notes

Large raw Anatel extracts should not be committed directly to GitHub. The repository should store reproducible acquisition notes and processed municipal-level files. Raw files can be regenerated or downloaded from official public sources and documented in `docs/DATA_AVAILABILITY.md`.

## Current Report

The latest interim report is stored in:

```text
reports/interim/QM640_Interim_Report_Four_RQs_APA7.pdf
reports/interim/QM640_Interim_Report_Four_RQs_APA7.docx
```
