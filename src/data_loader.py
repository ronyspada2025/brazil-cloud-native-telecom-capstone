"""Data acquisition and loading for the Brazil cloud-native telecom capstone.

All seven datasets are official open-government data products (Anatel / IBGE).
Because the portals serve some files behind interactive pages (and layouts can
change), this module deliberately does NOT hard-code direct file URLs that
could silently break or fetch the wrong artifact. Instead it:

1. documents the official landing page for each dataset
   (run:  python -m src.data_loader --instructions),
2. validates every raw file on load against an expected-column map, and
3. fails loudly with a clear message when a file or column is missing.

Place downloaded files in data/raw/ under the exact names in RAW_FILES.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

#: Official landing pages for the seven source datasets (verified 2026-07-23).
SOURCES: dict[str, dict[str, str]] = {
    "acessos": {
        "file": "anatel_acessos_raw.csv",
        "title": "Anatel - Acessos, Telefonia Movel (SMP)",
        "url": "https://dados.gov.br/dados/conjuntos-dados/acessos-autorizadas-smp",
        "notes": "Monthly accesses per provider/municipality/technology.",
    },
    "estacoes": {
        "file": "anatel_estacoes_raw.csv",
        "title": "Anatel - Outorga e Licenciamento, Estacoes Licenciadas (SMP)",
        "url": (
            "https://dados.gov.br/dados/conjuntos-dados/"
            "outorga-e-licenciamento---estaes-licenciadas"
        ),
        "notes": "Station-level ERB licensing records, updated daily.",
    },
    "rqual": {
        "file": "anatel_rqual_raw.csv",
        "title": "Anatel - Selos e indices de qualidade (RQUAL)",
        "url": (
            "https://www.gov.br/anatel/pt-br/dados/qualidade/"
            "qualidade-dos-servicos/selos-qualidade"
        ),
        "notes": "IQS/IQP/IR quality indicators and municipal quality seals.",
    },
    "slp": {
        "file": "anatel_slp_raw.csv",
        "title": "Anatel - Autorizadas do Servico Limitado Privado (SLP)",
        "url": (
            "https://www.anatel.gov.br/dadosabertos/"  # legacy catalog page decommissioned
        ),
        "notes": "Registry of entities authorized to operate private networks.",
    },
    "satisfacao": {
        "file": "anatel_satisfacao_raw.csv",
        "title": "Anatel - Pesquisa de Satisfacao e Qualidade Percebida",
        "url": (
            "https://www.gov.br/anatel/pt-br/consumidor/"
            "pesquisa-de-satisfacao-e-qualidade"
        ),
        "notes": "Interview-level microdata; download the mobile (SMP) file.",
    },
    "meu_municipio": {
        "file": "anatel_meu_municipio_raw.csv",
        "title": "Anatel - Meu Municipio (Acessos e Cobertura)",
        "url": "https://informacoes.anatel.gov.br/paineis/",  # legacy dataset page decommissioned
        "notes": "Municipal panorama incl. backhaul/fiber availability.",
    },
    "ibge": {
        "file": "ibge_socioeconomic_raw.csv",
        "title": "IBGE Cidades - PIB dos Municipios and demographics",
        "url": "https://cidades.ibge.gov.br",
        "notes": "Export municipal GDP, sectoral value added, population.",
    },
}

#: Minimal columns each raw file must contain after download.
#: Update these maps if the portals rename columns (notebook 01 will tell you).
EXPECTED_COLUMNS: dict[str, list[str]] = {
    # Fill in after first download, e.g.:
    # "acessos": ["Ano", "Mês", "Empresa", "Código IBGE Município", "Tecnologia
    #             Geração", "Acessos"],
    "acessos": [],
    "estacoes": [],
    "rqual": [],
    "slp": [],
    "satisfacao": [],
    "meu_municipio": [],
    "ibge": [],
}


#: The committed analysis input and its expected structure (final report, Table 3 / docs/DATA_PROVENANCE.md).
MERGED_FILE = PROCESSED_DIR / "merged_municipal_dataset.csv"
MERGED_SHA256 = "50fac84b16f63d66628741f36686cc076f43c90632a3befe53ca43ed9b316207"
MERGED_COLUMNS = [
    "MUNICIP_ID", "NOME", "PIB_MIL_REAIS", "POP_2024", "V93_populacao residente",
    "V6318_area da unidade terr", "V614_densidade demografic", "UF", "GDP_PER_CAP",
    "ACC_CDMA_IS_95", "ACC_GSM", "ACC_LTE", "ACC_NR", "ACC_WCDMA",
    "ERB_CDMA", "ERB_EDGE", "ERB_GSM", "ERB_LTE", "ERB_NR", "ERB_WCDMA",
    "SLP_STATION_CNT", "PRIVATE_5G_LIC", "AVG_DL_SPEED", "FIBER_ACCESSES", "FIBER_BACKHAUL",
]


def check_merged_dataset() -> bool:
    """Verify the committed analysis input: presence, 25 expected columns, and SHA-256."""
    import hashlib
    if not MERGED_FILE.exists():
        print(f"MISSING  {MERGED_FILE}")
        return False
    h = hashlib.sha256()
    with open(MERGED_FILE, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    cols = list(pd.read_csv(MERGED_FILE, nrows=0).columns)
    ok_cols = cols == MERGED_COLUMNS
    ok_sha = digest == MERGED_SHA256
    print(f"{'OK ' if ok_cols else 'BAD'}  columns   ({len(cols)} found; expected {len(MERGED_COLUMNS)})")
    print(f"{'OK ' if ok_sha else 'BAD'}  sha256    {digest}")
    return ok_cols and ok_sha


def print_instructions() -> None:
    """Print step-by-step download guidance for all seven datasets."""
    print("=" * 72)
    print("DOWNLOAD INSTRUCTIONS - official open-government sources only")
    print("=" * 72)
    for key, meta in SOURCES.items():
        print(f"\n[{key}] {meta['title']}")
        print(f"  1. Open: {meta['url']}")
        print("  2. Download the CSV/ZIP resource (extract if zipped).")
        print(f"  3. Save/rename to: data/raw/{meta['file']}")
        print(f"  Note: {meta['notes']}")
    print(
        "\nAfter placing all files, verify with:\n"
        "  python -m src.data_loader --check\n"
    )


def check_raw_files() -> bool:
    """Return True if all expected raw files exist; print a status table."""
    ok = True
    for key, meta in SOURCES.items():
        path = RAW_DIR / meta["file"]
        status = "OK " if path.exists() else "MISSING"
        if not path.exists():
            ok = False
        print(f"  [{status}] data/raw/{meta['file']}")
    return ok


def load_raw(key: str, **read_csv_kwargs) -> pd.DataFrame:
    """Load one raw dataset by key, validating expected columns.

    Parameters
    ----------
    key:
        One of: acessos, estacoes, rqual, slp, satisfacao, meu_municipio, ibge.
    read_csv_kwargs:
        Passed to :func:`pandas.read_csv` (e.g. ``sep=';'``,
        ``encoding='latin-1'`` - common for Anatel extracts).
    """
    if key not in SOURCES:
        raise KeyError(f"Unknown dataset key {key!r}. Valid: {list(SOURCES)}")
    path = RAW_DIR / SOURCES[key]["file"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python -m src.data_loader --instructions' "
            f"and download it from {SOURCES[key]['url']}"
        )
    df = pd.read_csv(path, **read_csv_kwargs)
    missing = [c for c in EXPECTED_COLUMNS.get(key, []) if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name}: expected columns missing {missing}. The portal may "
            "have restructured the file; update EXPECTED_COLUMNS in "
            "src/data_loader.py and the cleaning notebook accordingly."
        )
    return df


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Write a processed dataset to data/processed/ and return its path."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name
    df.to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instructions", action="store_true",
                        help="print download guidance for the 7 datasets")
    parser.add_argument("--check", action="store_true",
                        help="verify the committed merged dataset (columns + SHA-256) and list raw files")
    args = parser.parse_args(argv)
    if args.instructions:
        print_instructions()
        return 0
    if args.check:
        ok = check_merged_dataset()
        check_raw_files()
        return 0 if ok else 1
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
