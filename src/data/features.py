"""
Construction des features ML depuis le dataset DVF enrichi.
"""

import pandas as pd
import numpy as np

from src.data.load import load_dvf_joined
from src.utils.config import DATA_PROCESSED_DIR
from src.utils.logging import logger

ANNEE_REF = 2025

DPE_ORDINAL: dict[str, int] = {
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construire les features ML depuis le dataset DVF enrichi (après jointures).

    Features calculées :
    - surface_par_piece, copro_petite/grande
    - dpe_ordinal, dpe_bonus, dpe_malus (depuis etiquette_dpe)
    - age_bien (depuis annee_construction)
    - score_commodites
    - distance_tram_proche_m, distance_gare_proche_m
    - zone_inondable

    Note : prix_m2_median_quartier et nb_ventes_quartier sont calculés
    dans split.py sur le train set uniquement pour éviter le data leakage.
    """
    df = df.copy()

    df["surface_par_piece"] = df["surface_m2"] / df["nombre_pieces"].replace(0, np.nan)
    df["copro_petite"] = (df["nb_lots_copro"] <= 10).astype(int)
    df["copro_grande"] = (df["nb_lots_copro"] >= 50).astype(int)

    dpe_col = "etiquette_dpe" if "etiquette_dpe" in df.columns else None
    if dpe_col:
        df["dpe_ordinal"] = df[dpe_col].map(DPE_ORDINAL)
        df["dpe_bonus"] = df[dpe_col].isin(["A", "B"]).astype(int)
        df["dpe_malus"] = df[dpe_col].isin(["F", "G"]).astype(int)
    else:
        logger.warning("Colonne etiquette_dpe absente, features DPE ignorées")
        df["dpe_ordinal"] = np.nan
        df["dpe_bonus"] = 0
        df["dpe_malus"] = 0

    if "annee_construction" in df.columns:
        df["age_bien"] = ANNEE_REF - pd.to_numeric(df["annee_construction"], errors="coerce")
        df.loc[df["age_bien"] < 0, "age_bien"] = np.nan
    else:
        df["age_bien"] = np.nan

    commodite_cols = ["nb_commerces_500m", "nb_restaurants_500m", "nb_ecoles_500m", "nb_parcs_500m"]
    available = [c for c in commodite_cols if c in df.columns]
    if available:
        df["score_commodites"] = df[available].sum(axis=1)

    for col in ["distance_tram_proche_m", "distance_gare_proche_m"]:
        if col not in df.columns:
            df[col] = np.nan

    if "zone_inondable" not in df.columns:
        df["zone_inondable"] = np.nan

    return df


if __name__ == "__main__":
    df = load_dvf_joined()
    df = build_features(df)
    output = DATA_PROCESSED_DIR / "dvf_angers_features.parquet"
    df.to_parquet(output, index=False)
    logger.info(f"✓ Features construites : {df.shape} → {output.name}")
