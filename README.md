# Predicting Brazil's Cloud-Native Telecommunications Transformation

**A Municipal-Level Machine Learning Analysis of 5G Standalone Rollout, Service Quality, and Private Network Adoption**

- **Author:** Rony Anderson Spada Pedroso
- **Institution:** Walsh College — QM640: Data Analytics Capstone
- **Mentor:** Sridhar Srinivas

## Abstract

This doctoral capstone project evaluates, classifies, and predicts the progress and
operational drivers of the cloud-native telecommunications transformation in Brazil.
Using municipal-level open-government data from Anatel and IBGE, the study classifies
5G Standalone (SA) rollout paths, statistically compares service quality across
network generations, evaluates structural backhaul constraints, segments Brazilian
micro-regions by digital readiness, and models the drivers of private 5G (SLP)
adoption. The result is a predictive and descriptive analytical framework that
operators and regulators can use to optimize infrastructure investments.

## Research Questions

| RQ | Focus | Method |
|----|-------|--------|
| RQ1 | Can socioeconomic markers and existing 4G infrastructure predict high-density cloud-native 5G SA deployment? | Random forest classification |
| RQ2 | Do quality-of-service metrics differ across 4G, 5G NSA, and 5G SA networks? | One-way ANOVA + Tukey HSD |
| RQ3 | Which infrastructure characteristics best predict virtualized base station density? | Ridge regression (k = 6 predictors) |
| RQ4 | How do micro-regions cluster on digital readiness, and is membership independent of geography? | K-means + chi-square test |
| RQ5 | Can technical QoS metrics predict high customer-satisfaction classes? | XGBoost classification |
| RQ6 | What factors drive private 5G (SLP) license presence in a municipality? | Binary logistic regression |

## Data Sources (all official open-government data)

All seven datasets are authentic open-data products maintained by the Agência
Nacional de Telecomunicações (Anatel) and the Instituto Brasileiro de Geografia e
Estatística (IBGE). **No synthetic or fictional data are used.**

| # | Dataset | Local raw file | Official source |
|---|---------|----------------|-----------------|
| 1 | Acessos – Telefonia Móvel (SMP) | `data/raw/anatel_acessos_raw.csv` | https://dados.gov.br/dados/conjuntos-dados/acessos-autorizadas-smp |
| 2 | Outorga e Licenciamento – Estações Licenciadas | `data/raw/anatel_estacoes_raw.csv` | https://dados.gov.br/dados/conjuntos-dados/outorga-e-licenciamento---estaes-licenciadas |
| 3 | Selos e índices de qualidade (RQUAL) | `data/raw/anatel_rqual_raw.csv` | https://www.gov.br/anatel/pt-br/dados/qualidade/qualidade-dos-servicos/selos-qualidade |
| 4 | Autorizadas do Serviço Limitado Privado (SLP) | `data/raw/anatel_slp_raw.csv` | https://legado.dados.gov.br/dataset/autorizadas-do-servico-limitado-privado-slp |
| 5 | Pesquisa de Satisfação e Qualidade Percebida | `data/raw/anatel_satisfacao_raw.csv` | https://www.gov.br/anatel/pt-br/consumidor/pesquisa-de-satisfacao-e-qualidade |
| 6 | Meu Município – Acessos e Cobertura | `data/raw/anatel_meu_municipio_raw.csv` | https://dados.gov.br/dataset/meu-municipio-anatel |
| 7 | IBGE Cidades (PIB dos Municípios and demographics) | `data/raw/ibge_socioeconomic_raw.csv` | https://cidades.ibge.gov.br |

Raw files are **not committed** to the repository (see `.gitignore`) because of
their size. Run `python -m src.data_loader --instructions` for step-by-step
download guidance, then place the files in `data/raw/` under the exact names above.

## Repository Structure

```
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                      # source extracts (git-ignored; download locally)
│   └── processed/
│       ├── merged_municipal_dataset.csv
│       ├── quality_metrics_aggregated.csv
│       └── satisfaction_uf_provider.csv
├── notebooks/
│   ├── 01_data_cleaning_and_alignment.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_statistical_tests_rq2_rq4.ipynb
│   └── 05_ml_pipelines_rq1_rq3_rq5_rq6.ipynb
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── feature_engineering.py
    ├── model_evaluation.py
    └── visualization_utils.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.data_loader --instructions   # download guidance for the 7 datasets
jupyter lab                      # then run notebooks 01 -> 05 in order
```

## Reproducibility Notes

- The join key across all datasets is the 7-digit IBGE municipal code (`MUNICIP_ID`).
- Derived variables (`SPECTRUM_DENSITY`, `OPERATOR_SHARE`, `TECH_GENERATION`,
  `CSAT_CLASS`) are constructed in `notebooks/02_feature_engineering.ipynb`
  via `src/feature_engineering.py`.
- RQ5 is estimated at the provider–federative-unit grain
  (`data/processed/satisfaction_uf_provider.csv`), consistent with the survey's
  geographic identification.
- Portals occasionally restructure their CSV layouts. Notebook 01 validates
  expected columns on load and fails loudly with a clear message if a column
  is missing — update the column maps in `src/data_loader.py` if that happens.

## License

Released under the MIT License (see `LICENSE`). The underlying government data
are open data published by Anatel and IBGE under their respective open-data
policies; please credit the original sources.
