"""
Split train/test du dataset final.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load import load_dvf_features
from src.utils.config import DATA_PROCESSED_DIR, settings
from src.utils.logging import logger

# Colonnes utilisées comme features (enrichi progressivement)
BASE_FEATURES = [
    "surface_m2",
    "nombre_pieces",
    "surface_par_piece",
    "nb_lots_copro",
    "mois_vente",
    "annee_vente",
    "copro_petite",
    "copro_grande",
]

OPTIONAL_FEATURES = [
    "dpe_ordinal",
    "dpe_bonus",
    "dpe_malus",
    "age_bien",
    "prix_m2_median_quartier",
    "nb_ventes_quartier",
    "prix_m2_vs_quartier",
    "nb_commerces_500m",
    "nb_restaurants_500m",
    "nb_ecoles_500m",
    "nb_parcs_500m",
    "score_commodites",
]

TARGET = "prix_m2"


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retourner les colonnes de features disponibles dans le DataFrame."""
    return [col for col in BASE_FEATURES + OPTIONAL_FEATURES if col in df.columns]


def split_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Charger le dataset final, sélectionner les features et splitter train/test.

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = load_dvf_features()
    feature_cols = get_feature_columns(df)

    df_clean = df[feature_cols + [TARGET]].dropna()
    X = df_clean[feature_cols]
    y = df_clean[TARGET]

    logger.info(f"Features utilisées ({len(feature_cols)}) : {feature_cols}")
    logger.info(f"Échantillons disponibles : {len(X):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.test_size, random_state=settings.random_seed
    )

    # Sauvegarder les splits
    split_dir = DATA_PROCESSED_DIR / "train_test_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(split_dir / "X_train.parquet", index=False)
    X_test.to_parquet(split_dir / "X_test.parquet", index=False)
    y_train.to_frame().to_parquet(split_dir / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(split_dir / "y_test.parquet", index=False)

    logger.info(f"✓ Split sauvegardé : train={len(X_train):,} | test={len(X_test):,}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    split_dataset()
