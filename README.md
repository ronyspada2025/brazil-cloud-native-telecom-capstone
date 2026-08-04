# QM640 Data Analytics Capstone

## The State of Cloud-Native Transformation on Telecommunications Networks in Brazil

Author: Rony Anderson Spada Pedroso  
Institution: Walsh College  
Course: QM640: Data Analytics Capstone  

## Repository Purpose

This repository contains the data-processing, exploratory data analysis, and reporting assets for the capstone project on Brazil's municipal-level cloud-native telecommunications transformation. The project uses open public data from Anatel and IBGE to evaluate 5G SA/NR rollout, fiber-readiness intensity, service-quality indicators, digital-readiness patterns, and private-network adoption.

## Revised Repository Directory Structure

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
│   └── capstone_synopsis_brazil_telecom_RONY_PEDROSO.pdf
├── notebooks/
│   ├── 01_data_understanding_and_cleaning.ipynb
│   ├── 02_complete_eda.ipynb
│   └── 03_rq_exploratory_analysis.ipynb
├── reports/
│   ├── interim/
│   │   ├── QM640_Interim_Report_APA7_Revised.docx
│   │   └── QM640_Interim_Report_APA7_Revised.pdf
│   ├── final/
│   └── figures/
├── src/
└── tests/
```

## Notebook Execution Order

1. `notebooks/01_data_understanding_and_cleaning.ipynb`
2. `notebooks/02_complete_eda.ipynb`
3. `notebooks/03_rq_exploratory_analysis.ipynb`

## Research Questions

- RQ1: Can municipal socioeconomic markers and existing infrastructure predict high-density 5G SA deployment?
- RQ2: Is there a statistically significant difference in service-quality metrics across network generations?
- RQ3: Which infrastructure characteristics are the strongest predictors of virtualized base-station density?
- RQ4: How do municipalities cluster by digital-transformation readiness, and are clusters independent of region?
- RQ5: Can technical network indicators predict high customer satisfaction?
- RQ6: Which industrial, agricultural, and infrastructure factors influence private 5G/SLP presence?

## Data Availability

The processed municipal dataset is stored under `data/processed/`. Large raw Anatel extracts should not be committed directly to GitHub. They should be reproduced through official public sources and documented acquisition notebooks.

## Environment Setup

```bash
pip install -r requirements.txt
```

## Notes

The interim EDA identified duplicate-key merge inflation in the uploaded dataset and corrected the analysis frame to 5,571 municipalities. The EDA also found that binary fiber presence is saturated and that fiber intensity is more analytically useful.
