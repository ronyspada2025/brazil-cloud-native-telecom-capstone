"""Construction of the derived variables defined in Table 2 of the synopsis.

Derived variables
-----------------
SPECTRUM_DENSITY : count of distinct licensed frequency bands per municipality
                   (from the station records' transmission-frequency field).
OPERATOR_SHARE   : leading operator's share of municipal mobile accesses (%).
TECH_GENERATION  : dominant network generation (4G / 5G NSA / 5G SA) of a
                   provider-municipality record, used as the RQ2 grouping.
CSAT_CLASS       : binary high-satisfaction class (1 = ISG >= 7.0).

Each function takes cleaned dataframes produced by notebook 01 and returns a
municipal (or provider-municipality) frame keyed by MUNICIP_ID. Column names
referencing the raw extracts are parameters so the functions survive portal
renames - set them once in notebook 02.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: 5G SA-capable band heuristic (MHz): 3.5 GHz auction band. Adjust with care
#: and cross-validate against Anatel's station panel technology field.
SA_BAND_RANGE_MHZ = (3300.0, 3700.0)

CSAT_THRESHOLD = 7.0


def spectrum_density(stations: pd.DataFrame,
                     municip_col: str = "MUNICIP_ID",
                     freq_col: str = "FreqTxMHz",
                     band_width_mhz: float = 100.0) -> pd.DataFrame:
    """SPECTRUM_DENSITY: distinct frequency bands licensed per municipality.

    Frequencies are bucketed into ``band_width_mhz`` bands before counting so
    adjacent channel assignments within one band are not double-counted.
    """
    df = stations[[municip_col, freq_col]].dropna().copy()
    df["band"] = (df[freq_col] // band_width_mhz).astype(int)
    out = (df.groupby(municip_col)["band"].nunique()
             .rename("SPECTRUM_DENSITY").reset_index())
    return out


def operator_share(accesses: pd.DataFrame,
                   municip_col: str = "MUNICIP_ID",
                   operator_col: str = "Empresa",
                   accesses_col: str = "Acessos") -> pd.DataFrame:
    """OPERATOR_SHARE: leading operator's share (%) of municipal accesses."""
    g = accesses.groupby([municip_col, operator_col])[accesses_col].sum()
    total = g.groupby(level=0).sum()
    lead = g.groupby(level=0).max()
    out = (100.0 * lead / total).rename("OPERATOR_SHARE").reset_index()
    return out


def tech_generation(accesses: pd.DataFrame,
                    municip_col: str = "MUNICIP_ID",
                    operator_col: str = "Empresa",
                    tech_col: str = "TecnologiaGeracao",
                    accesses_col: str = "Acessos",
                    sa_label: str = "5G-SA",
                    nsa_label: str = "5G-NSA") -> pd.DataFrame:
    """TECH_GENERATION: dominant generation per provider-municipality pair.

    Returns one row per (municipality, operator) with the generation holding
    the largest access count, mapped onto {'4G', '5G NSA', '5G SA'}. Records
    dominated by pre-4G technologies are labeled '4G' only if 4G is the
    largest post-3G share; otherwise they are dropped (out of RQ2 scope).
    """
    mapping = {sa_label: "5G SA", nsa_label: "5G NSA", "4G": "4G"}
    df = accesses[accesses[tech_col].isin(mapping)].copy()
    df["gen"] = df[tech_col].map(mapping)
    idx = (df.groupby([municip_col, operator_col, "gen"])[accesses_col]
             .sum().reset_index())
    top = (idx.sort_values(accesses_col, ascending=False)
              .drop_duplicates([municip_col, operator_col]))
    return top.rename(columns={"gen": "TECH_GENERATION"})[
        [municip_col, operator_col, "TECH_GENERATION"]
    ]


def csat_class(survey: pd.DataFrame,
               isg_col: str = "ISG",
               threshold: float = CSAT_THRESHOLD) -> pd.DataFrame:
    """CSAT_CLASS: 1 if ISG >= threshold (default 7.0) else 0."""
    out = survey.copy()
    out["CSAT_CLASS"] = (out[isg_col].astype(float) >= threshold).astype(int)
    return out


def dominant_generation_share(accesses: pd.DataFrame,
                              municip_col: str = "MUNICIP_ID",
                              tech_col: str = "TecnologiaGeracao",
                              accesses_col: str = "Acessos") -> pd.DataFrame:
    """Share of the dominant generation per municipality (RQ2 sensitivity).

    Used to restrict the RQ2 sensitivity analysis to municipalities where a
    single generation clearly predominates (e.g., share >= 0.70).
    """
    g = accesses.groupby([municip_col, tech_col])[accesses_col].sum()
    total = g.groupby(level=0).sum()
    lead = g.groupby(level=0).max()
    out = (lead / total).rename("DOMINANT_GEN_SHARE").reset_index()
    return out
