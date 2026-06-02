"""
Chargement des fichiers de données brutes et processées.
"""

import pandas as pd

from src.utils.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, DVF_YEARS, settings
from src.utils.logging import logger


def load_dvf_raw(years: list[int] | None = None) -> pd.DataFrame:
    """
    Charger les fichiers DVF bruts pour les années demandées.

    Args:
        years: Années à charger (défaut : DVF_YEARS depuis config)

    Returns:
        DataFrame concaténé de toutes les années disponibles
    """
    years = years or DVF_YEARS
    dvf_dir = DATA_RAW_DIR / "dvf"
    dfs: list[pd.DataFrame] = []

    for year in years:
        filepath = dvf_dir / f"dvf_{year}_{settings.department_code}.csv"
        if not filepath.exists():
            logger.warning(f"Fichier DVF {year} introuvable : {filepath.name}")
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
        raise FileNotFoundError("Aucun fichier DVF trouvé. Lancer : make data-download")

    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"DVF brut chargé : {len(result):,} lignes")
    return result


def load_dvf_clean() -> pd.DataFrame:
    """Charger le fichier DVF nettoyé."""
    path = DATA_PROCESSED_DIR / "dvf_angers_appart_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} introuvable. Lancer : make data-clean")
    return pd.read_parquet(path)


def load_dvf_joined() -> pd.DataFrame:
    """Charger le fichier DVF enrichi (après jointures DPE + OSM + IRIS)."""
    path = DATA_PROCESSED_DIR / "dvf_angers_joined.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} introuvable. Lancer : make data-join")
    return pd.read_parquet(path)


def load_dvf_features() -> pd.DataFrame:
    """Charger le fichier DVF avec toutes les features ML."""
    path = DATA_PROCESSED_DIR / "dvf_angers_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} introuvable. Lancer : make data-features")
    return pd.read_parquet(path)


def load_dpe() -> pd.DataFrame:
    """Charger les données DPE ADEME."""
    path = DATA_RAW_DIR / "dpe" / "dpe_angers.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} introuvable. Lancer : make data-download")
    return pd.read_parquet(path)


def load_equipements_osm() -> pd.DataFrame:
    """Charger les équipements OSM (Overpass API)."""
    path = DATA_RAW_DIR / "osm" / "equipements_osm.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} introuvable. Lancer : make data-download")
    return pd.read_parquet(path)
