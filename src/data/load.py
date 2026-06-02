"""
Chargement des fichiers de données brutes et processées.
"""

import pandas as pd

from src.utils.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, settings
from src.utils.logging import logger


def load_dvf_raw(years: list[int] | None = None) -> pd.DataFrame:
    """
    Charger les fichiers DVF bruts pour les années demandées.

    Args:
        years: Années à charger (défaut : config)

    Returns:
        DataFrame concaténé de toutes les années
    """
    years = years or settings.dvf_years
    dvf_dir = DATA_RAW_DIR / "dvf"
    dfs: list[pd.DataFrame] = []

    for year in years:
        filepath = dvf_dir / f"dvf_{year}_{settings.department_code}.csv"
        if not filepath.exists():
            logger.warning(f"Fichier DVF {year} introuvable : {filepath}")
            continue
        logger.info(f"Chargement {filepath.name}...")
        df = pd.read_csv(
            filepath,
            dtype={"code_commune": str, "code_parcelle": str},
            parse_dates=["date_mutation"],
            low_memory=False,
        )
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("Aucun fichier DVF trouvé. Lancer make data-download.")

    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"DVF brut chargé : {len(result):,} lignes")
    return result


def load_dvf_clean() -> pd.DataFrame:
    """Charger le fichier DVF nettoyé."""
    path = DATA_PROCESSED_DIR / "dvf_angers_appart_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable. Lancer make data-clean.")
    return pd.read_parquet(path)


def load_dvf_features() -> pd.DataFrame:
    """Charger le fichier DVF avec toutes les features."""
    path = DATA_PROCESSED_DIR / "dvf_angers_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable. Lancer make data-features.")
    return pd.read_parquet(path)


def load_dpe() -> pd.DataFrame:
    """Charger les données DPE Ademe."""
    path = DATA_RAW_DIR / "dpe" / "dpe_france.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable. Lancer make data-download.")
    return pd.read_parquet(path)


def load_bpe() -> pd.DataFrame:
    """Charger la Base Permanente des Équipements INSEE."""
    path = DATA_RAW_DIR / "bpe" / "bpe_insee.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable. Lancer make data-download.")
    return pd.read_csv(path, dtype={"depcom": str, "dciris": str}, low_memory=False)


def load_iris() -> pd.DataFrame:
    """Charger la table d'appartenance géographique des IRIS INSEE."""
    path = DATA_RAW_DIR / "iris" / "iris_insee.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable. Lancer make data-download.")
    return pd.read_csv(path, dtype={"CODE_IRIS": str, "DEP": str}, low_memory=False)


def load_copro() -> pd.DataFrame:
    """Charger les données copropriétés."""
    path = DATA_RAW_DIR / "copro" / "copro_api.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable. Lancer make data-download.")
    return pd.read_csv(path, low_memory=False)
