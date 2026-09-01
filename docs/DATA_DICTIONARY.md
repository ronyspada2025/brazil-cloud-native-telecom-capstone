# Data Dictionary — Final Report (Modeled Variables)

Analysis input: `data/processed/merged_municipal_dataset.csv` (10,107 × 25 as supplied; 5,571 × 42 after `final_pipeline.py` cleaning and feature engineering). Every variable below is either a column of the supplied merged dataset or a deterministic transformation of such columns; no external variables were introduced.

| Variable | Definition | Type | Role |
|---|---|---|---|
| MUNICIP_ID | 7-digit IBGE municipal code (string) | Key | Join key |
| POP_2024 | Estimated resident population, 2024 | Continuous | Predictor (RQ1–RQ4) |
| POP_DENSITY | Population per km², Census 2022 (renamed from `V614_densidade demografic`) | Continuous | Predictor (RQ1–RQ4) |
| AREA_KM2 | Territorial area (renamed from `V6318_area da unidade terr`) | Continuous | Feature engineering |
| POP_CENSUS_2022 | Census 2022 resident population (renamed from `V93_populacao residente`) | Continuous | Reference |
| GDP_PER_CAP | Municipal GDP per capita (BRL) | Continuous | Predictor (RQ1–RQ4) |
| FIBER_PER_100 | Fiber broadband accesses per 100 inhabitants (engineered from FIBER_ACCESSES / POP_2024) | Continuous | Predictor (RQ1–RQ4); preferred over the saturated FIBER_BACKHAUL flag |
| LTE_ACCESS_PER_100 | LTE accesses per 100 inhabitants | Continuous | Predictor (RQ1–RQ4) |
| REGION | Macro-region decoded from first digit of MUNICIP_ID (1=North, 2=Northeast, 3=Southeast, 4=South, 5=Center-West) | Categorical | Predictor dummies; RQ3 association test |
| NR_STATION_CNT | Licensed NR radio-station count (renamed from `ERB_NR`); a station-side proxy that does not establish SA-core or cloud-native operation | Integer | RQ1 target construction; RQ2 count-model outcome |
| HIGH_NR_P75 | 1 if NR_STATION_CNT > 9 (75th percentile); a raw-count class that partly encodes municipal scale; P70 = 6 and P80 = 12 used for sensitivity | Binary | RQ1 primary target |
| HIGH_NR_PER_100K_P75 | 1 if NR_PER_100K_POP > 46.4 (75th percentile) | Binary | RQ1 per-capita robustness target |
| NR_PER_100K_POP | NR stations per 100,000 population | Continuous | RQ2 target |
| NR_ACCESS_PER_100 | NR accesses per 100 inhabitants | Continuous | RQ3 feature; **excluded from RQ1/RQ2 (leakage)** |
| AVG_DL_SPEED | Mean measured download speed (Mbps) | Continuous | EDA outcome; RQ3 feature; **excluded from RQ1/RQ2 (leakage)** |
| SLP_STATION_CNT | Private-network (SLP) station count; the only field that differs between the two rows a municipality receives from the one-to-many SLP-registry join, so it is summed on duplicate-key collapse | Integer | RQ4 target construction |
| SLP_PER_100K_POP | SLP stations per 100,000 population | Continuous | RQ4 target construction |
| HIGH_SLP | 1 if SLP_PER_100K_POP > 1,618 (75th percentile); private-network-related activity, not private 5G specifically | Binary | RQ4 target |
| PRIVATE_5G_LIC | SLP presence flag (97.8% = 1; saturated) | Binary | Descriptive only |
| LOG1P_* | log1p transformations of the skewed continuous variables above | Continuous | Modeling/visualization |
