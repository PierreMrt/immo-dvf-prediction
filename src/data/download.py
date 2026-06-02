"""
Téléchargement des données DVF, DPE, BPE, IRIS, COPRO depuis data.gouv.fr / data.ademe.fr.
"""

import requests
from pathlib import Path

from src.utils.config import DATA_RAW_DIR, DVF_YEARS, settings
from src.utils.logging import logger

# UUIDs des ressources sur data.gouv.fr (mettre à jour si besoin)
# https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres-geolocalisees/
DVF_UUIDS: dict[int, str] = {
    2024: "<uuid-dvf-geolocalisees-2024>",
    2025: "<uuid-dvf-geolocalisees-2025>",
    2026: "<uuid-dvf-geolocalisees-2026>",
}

DATA_GOUV_BASE = "https://www.data.gouv.fr/fr/datasets/r/"


def _download_file(url: str, dest: Path, timeout: int = 600) -> None:
    """Télécharger un fichier en streaming depuis une URL."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def download_dvf_geolocalisees(years: list[int] | None = None) -> None:
    """
    Télécharger les données DVF géolocalisées pour le département 49.

    Args:
        years: Années à télécharger (défaut : config)
    """
    years = years or DVF_YEARS
    dvf_dir = DATA_RAW_DIR / "dvf"

    for year in years:
        if year not in DVF_UUIDS:
            logger.warning(f"Année DVF {year} non référencée, saut")
            continue

        dest = dvf_dir / f"dvf_{year}_{settings.department_code}.csv"
        if dest.exists():
            logger.info(f"DVF {year} déjà présent, saut ({dest})")
            continue

        logger.info(f"Téléchargement DVF {year}...")
        try:
            _download_file(f"{DATA_GOUV_BASE}{DVF_UUIDS[year]}", dest)
            logger.info(f"✓ DVF {year} téléchargé : {dest}")
        except requests.RequestException as e:
            logger.error(f"✗ Erreur DVF {year} : {e}")


def download_dpe() -> None:
    """Télécharger les données DPE Ademe (format parquet)."""
    dest = DATA_RAW_DIR / "dpe" / "dpe_france.parquet"
    if dest.exists():
        logger.info(f"DPE déjà présent, saut ({dest})")
        return

    url = "https://data.ademe.fr/datasets/dpe-france/files/download"
    logger.info("Téléchargement DPE Ademe...")
    try:
        _download_file(url, dest)
        logger.info(f"✓ DPE téléchargé : {dest}")
    except requests.RequestException as e:
        logger.error(f"✗ Erreur DPE : {e}")


def download_bpe() -> None:
    """Télécharger la Base Permanente des Équipements (BPE) INSEE."""
    dest = DATA_RAW_DIR / "bpe" / "bpe_insee.csv"
    if dest.exists():
        logger.info(f"BPE déjà présent, saut ({dest})")
        return

    url = f"{DATA_GOUV_BASE}<uuid-bpe-insee>"
    logger.info("Téléchargement BPE INSEE...")
    try:
        _download_file(url, dest)
        logger.info(f"✓ BPE téléchargé : {dest}")
    except requests.RequestException as e:
        logger.error(f"✗ Erreur BPE : {e}")


def download_iris() -> None:
    """Télécharger la table d'appartenance géographique des IRIS (INSEE)."""
    dest = DATA_RAW_DIR / "iris" / "iris_insee.csv"
    if dest.exists():
        logger.info(f"IRIS déjà présent, saut ({dest})")
        return

    url = f"{DATA_GOUV_BASE}<uuid-iris-insee>"
    logger.info("Téléchargement IRIS INSEE...")
    try:
        _download_file(url, dest)
        logger.info(f"✓ IRIS téléchargé : {dest}")
    except requests.RequestException as e:
        logger.error(f"✗ Erreur IRIS : {e}")


def download_copro() -> None:
    """Télécharger les données copropriétés (API COPRO data.gouv.fr)."""
    dest = DATA_RAW_DIR / "copro" / "copro_api.csv"
    if dest.exists():
        logger.info(f"COPRO déjà présent, saut ({dest})")
        return

    url = f"{DATA_GOUV_BASE}<uuid-copro>"
    logger.info("Téléchargement COPRO...")
    try:
        _download_file(url, dest)
        logger.info(f"✓ COPRO téléchargé : {dest}")
    except requests.RequestException as e:
        logger.error(f"✗ Erreur COPRO : {e}")


def download_all() -> None:
    """Télécharger toutes les sources de données."""
    download_dvf_geolocalisees()
    download_dpe()
    download_bpe()
    download_iris()
    download_copro()


if __name__ == "__main__":
    download_all()
