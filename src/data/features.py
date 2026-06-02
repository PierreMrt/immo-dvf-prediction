"""
Feature engineering : création des variables dérivées pour le modèle ML.
"""

import numpy as np
import pandas as pd

from src.utils.config import DATA_PROCESSED_DIR
from src.utils.logging import logger

ANNEE_REF = 2026

# Mapping classe énergie vers entier ordinal (A=1, G=7)
DPE_ORDINAL: dict[str, int] = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Créer les features pour la prédiction de prix.

    Args:
        df: DataFrame nettoyé (DVF + jointures)

    Returns:
        DataFrame enrichi avec toutes les features
    """
    df = df.copy()
    logger.info("Création des features...")

    # --- Features générales ---
    df["surface_par_piece"] = df["surface_m2"] / df["nombre_pieces"]
    df["log_prix_m2"] = np.log(df["prix_m2"])

    # --- Features copropriété ---
    df["copro_petite"] = (df["nb_lots_copro"] < 10).astype(int)
    df["copro_grande"] = (df["nb_lots_copro"] > 50).astype(int)

    # --- Features DPE (si disponibles après jointure) ---
    if "classe_energie" in df.columns:
        df["dpe_ordinal"] = df["classe_energie"].map(DPE_ORDINAL)
        df["dpe_bonus"] = df["classe_energie"].isin(["A", "B"]).astype(int)
        df["dpe_malus"] = df["classe_energie"].isin(["F", "G"]).astype(int)

    if "annee_construction" in df.columns:
        df["age_bien"] = (ANNEE_REF - df["annee_construction"]).clip(lower=0)

    # --- Features quartier IRIS (si disponibles après jointure) ---
    if "code_iris" in df.columns:
        iris_stats = (
            df.groupby("code_iris")["prix_m2"]
            .agg(prix_m2_median_quartier="median", nb_ventes_quartier="count")
            .reset_index()
        )
        df = df.merge(iris_stats, on="code_iris", how="left")
        df["prix_m2_vs_quartier"] = (
            (df["prix_m2"] - df["prix_m2_median_quartier"])
            / df["prix_m2_median_quartier"]
            * 100
        )

    # --- Score global commodités ---
    commodite_cols = [c for c in df.columns if c.endswith("_500m")]
    if commodite_cols:
        df["score_commodites"] = df[commodite_cols].sum(axis=1)

    logger.info(f"Features créées : {len(df.columns)} colonnes au total")
    return df


if __name__ == "__main__":
    input_path = DATA_PROCESSED_DIR / "dvf_angers_joined.parquet"
    df = pd.read_parquet(input_path)
    df_features = create_features(df)

    output_path = DATA_PROCESSED_DIR / "dvf_angers_features.parquet"
    df_features.to_parquet(output_path, index=False)
    logger.info(f"✓ Features sauvegardées : {output_path}")
