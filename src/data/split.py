"""
Split train/test du dataset final.

Les stats par quartier (prix_m2_median_quartier, nb_ventes_quartier) sont
calculées uniquement sur le train set pour éviter le data leakage, puis
jointes sur train et test. Le mapping iris_stats.parquet est sauvegardé
pour enrichir les nouvelles annonces dans l'app.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load import load_dvf_features
from src.utils.config import DATA_PROCESSED_DIR, settings
from src.utils.logging import logger

# Colonnes utilisées comme features
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
    # DPE
    "dpe_ordinal",
    "dpe_bonus",
    "dpe_malus",
    "age_bien",
    # Quartier — alimentées après split, sans leakage
    "prix_m2_median_quartier",
    "nb_ventes_quartier",
    # Équipements OSM
    "nb_commerces_500m",
    "nb_restaurants_500m",
    "nb_ecoles_500m",
    "nb_parcs_500m",
    "score_commodites",
    # Transports
    "distance_tram_proche_m",
    "distance_gare_proche_m",
    # Risques
    "zone_inondable",
]

TARGET = "prix_m2"


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retourner les colonnes de features disponibles dans le DataFrame."""
    return [col for col in BASE_FEATURES + OPTIONAL_FEATURES if col in df.columns]


def _compute_iris_stats(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame | None:
    """
    Calculer les stats par IRIS sur le train set uniquement.

    Returns:
        DataFrame avec code_iris, prix_m2_median_quartier, nb_ventes_quartier
        ou None si code_iris absent du train set.
    """
    if "code_iris" not in X_train.columns:
        return None
    train_with_target = X_train[["code_iris"]].copy()
    train_with_target[TARGET] = y_train.values
    stats = (
        train_with_target.groupby("code_iris")[TARGET]
        .agg(prix_m2_median_quartier="median", nb_ventes_quartier="count")
        .reset_index()
    )
    return stats


def split_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Charger le dataset final, sélectionner les features et splitter train/test.
    Les stats IRIS sont calculées sur le train set puis jointes sur les deux splits.

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = load_dvf_features()

    # Garder code_iris pour le calcul des stats, même si pas une feature finale
    has_iris = "code_iris" in df.columns
    feature_cols = get_feature_columns(df)
    cols_to_keep = list(dict.fromkeys(feature_cols + (["code_iris"] if has_iris else []) + [TARGET]))
    df_clean = df[cols_to_keep].dropna(subset=BASE_FEATURES + [TARGET])

    X = df_clean[feature_cols + (["code_iris"] if has_iris else [])]
    y = df_clean[TARGET]

    logger.info(f"Features utilisées ({len(feature_cols)}) : {feature_cols}")
    logger.info(f"Échantillons disponibles : {len(X):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.test_size, random_state=settings.random_seed
    )

    # Stats IRIS calculées sur train uniquement → pas de leakage
    iris_stats = _compute_iris_stats(X_train, y_train)
    if iris_stats is not None:
        # Sauvegarde pour l'app (enrichissement nouvelles annonces)
        split_dir = DATA_PROCESSED_DIR / "train_test_split"
        split_dir.mkdir(parents=True, exist_ok=True)
        iris_stats.to_parquet(split_dir / "iris_stats.parquet", index=False)
        logger.info(f"✓ iris_stats.parquet sauvegardé ({len(iris_stats)} zones IRIS)")

        for df_split in [X_train, X_test]:
            idx = df_split.index
            merged = df_split.merge(iris_stats, on="code_iris", how="left")
            merged.index = idx
            if "prix_m2_median_quartier" not in X_train.columns:
                X_train = X_train.merge(iris_stats, on="code_iris", how="left")
                X_train.index = X_train.index  # reset après merge
                break

        X_train = X_train.merge(iris_stats, on="code_iris", how="left") if "prix_m2_median_quartier" not in X_train.columns else X_train
        X_test = X_test.merge(iris_stats, on="code_iris", how="left") if "prix_m2_median_quartier" not in X_test.columns else X_test

    # Retirer code_iris des features finales (identifiant, pas une feature ML)
    X_train = X_train.drop(columns=["code_iris"], errors="ignore")
    X_test = X_test.drop(columns=["code_iris"], errors="ignore")

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
