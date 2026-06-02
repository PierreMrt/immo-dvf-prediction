"""
Téléchargement des données DVF, DPE et BPE depuis leurs URLs stables.

Sources :
- DVF géolocalisées : https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/{dep}.csv.gz
  Disponibles avec un décalage : l'année N est publiée courant N+1.
- DPE (depuis juil. 2021) : https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant
- BPE 2024 avec coord. : https://www.insee.fr/fr/statistiques/fichier/8217525/bpe24_ensemble_xy_csv.zip
  (page source : https://www.insee.fr/fr/statistiques/8217525?sommaire=8217537)
"""

import gzip
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import DATA_RAW_DIR, DVF_YEARS, settings
from src.utils.logging import logger

DVF_BASE_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/{dep}.csv.gz"
BPE_URL = "https://www.insee.fr/fr/statistiques/fichier/8217525/bpe24_ensemble_xy_csv.zip"
DPE_API_URL = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines"
DPE_BATCH_SIZE = 10_000

# Colonnes à récupérer depuis le dataset dpe03existant
# Référence schema : curl https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines?size=1
DPE_SELECT_COLS = ",".join([
    "numero_dpe",
    "date_etablissement_dpe",
    "adresse_ban",
    "code_postal_ban",
    "nom_commune_ban",
    "etiquette_dpe",                       # classe énergie (A-G)
    "etiquette_ges",
    "annee_construction",
    "surface_habitable_immeuble",           # surface disponible dans dpe03existant
    "consommation_energie_primaire",
    "emission_ges_5_usages_par_m2",         # émissions GES par m²
    "conso_5_usages_par_m2_ep",            # conso énergie primaire par m²
    "coordonnee_cartographique_x_ban",     # Lambert-93 X
    "coordonnee_cartographique_y_ban",     # Lambert-93 Y
    "periode_construction",
    "type_batiment",
])


def _download_stream(url: str, dest: Path, timeout: int = 600) -> None:
    """Télécharger un fichier en streaming vers dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65_536):
                f.write(chunk)


def download_dvf_geolocalisees(years: list[int] | None = None) -> None:
    """
    Télécharger les DVF géolocalisées pour le département configuré.
    Les fichiers .csv.gz sont décompressés automatiquement en .csv.

    Note : les DVF sont publiées avec un décalage d'environ 6 mois.
    L'année N n'est généralement disponible que courant N+1.
    """
    years = years or DVF_YEARS
    dep = settings.department_code
    dvf_dir = DATA_RAW_DIR / "dvf"

    for year in years:
        dest_csv = dvf_dir / f"dvf_{year}_{dep}.csv"
        if dest_csv.exists():
            logger.info(f"DVF {year} déjà présent, saut ({dest_csv.name})")
            continue

        url = DVF_BASE_URL.format(annee=year, dep=dep)
        dest_gz = dvf_dir / f"dvf_{year}_{dep}.csv.gz"
        logger.info(f"Téléchargement DVF {year} (département {dep})...")

        try:
            _download_stream(url, dest_gz)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                logger.warning(
                    f"⚠ DVF {year} non disponible (publiée avec décalage, réessayer plus tard)"
                )
            else:
                logger.error(f"✗ Erreur DVF {year} : {e}")
            if dest_gz.exists():
                dest_gz.unlink()
            continue
        except requests.RequestException as e:
            logger.error(f"✗ Erreur DVF {year} : {e}")
            if dest_gz.exists():
                dest_gz.unlink()
            continue

        with gzip.open(dest_gz, "rb") as f_in, open(dest_csv, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        dest_gz.unlink()
        logger.info(f"✓ DVF {year} téléchargé et décompressé : {dest_csv.name}")


def download_dpe(code_postal_prefix: str = "490") -> None:
    """
    Télécharger les DPE (depuis juillet 2021) via l'API ADEME par pagination.
    Filtre sur le code postal pour ne garder qu'Angers et environs.
    Dataset : dpe03existant
    """
    dest = DATA_RAW_DIR / "dpe" / "dpe_angers.parquet"
    if dest.exists():
        logger.info(f"DPE déjà présent, saut ({dest.name})")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Téléchargement DPE ADEME (code postal {code_postal_prefix}*)...")

    params: dict = {
        "size": DPE_BATCH_SIZE,
        "q": code_postal_prefix,
        "q_fields": "code_postal_ban",
        "select": DPE_SELECT_COLS,
    }

    dfs: list[pd.DataFrame] = []
    after = None
    page = 1

    while True:
        if after:
            params["after"] = after
        try:
            r = requests.get(DPE_API_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
        except requests.HTTPError as e:
            logger.error(f"✗ Erreur DPE page {page} ({e.response.status_code if e.response else '?'}) : {e}")
            break
        except requests.RequestException as e:
            logger.error(f"✗ Erreur DPE page {page} : {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        dfs.append(pd.DataFrame(results))
        logger.info(f"  Page {page} : {len(results)} enregistrements")
        after = data.get("next")
        if not after:
            break
        page += 1

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df.to_parquet(dest, index=False)
        logger.info(f"✓ DPE téléchargé : {len(df):,} lignes → {dest.name}")
    else:
        logger.warning("⚠ Aucune donnée DPE récupérée")


def download_bpe() -> None:
    """
    Télécharger la BPE 2024 (avec coordonnées Lambert-93) depuis l'INSEE.
    Filtre sur le département configuré après extraction du ZIP.
    Source : https://www.insee.fr/fr/statistiques/8217525?sommaire=8217537
    """
    dest = DATA_RAW_DIR / "bpe" / "bpe_insee.csv"
    if dest.exists():
        logger.info(f"BPE déjà présent, saut ({dest.name})")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest_zip = DATA_RAW_DIR / "bpe" / "bpe24_ensemble_xy.zip"

    logger.info("Téléchargement BPE 2024 INSEE...")
    try:
        _download_stream(BPE_URL, dest_zip)
    except requests.RequestException as e:
        logger.error(f"✗ Erreur BPE : {e}")
        return

    with zipfile.ZipFile(dest_zip, "r") as z:
        csv_files = [f for f in z.namelist() if f.endswith(".csv")]
        if not csv_files:
            logger.error("✗ Aucun CSV trouvé dans le ZIP BPE")
            dest_zip.unlink()
            return
        with z.open(csv_files[0]) as f:
            df = pd.read_csv(f, sep=";", dtype={"DEPCOM": str, "DEP": str}, low_memory=False)

    dest_zip.unlink()
    df_dep = df[df["DEP"] == settings.department_code].reset_index(drop=True)
    df_dep.to_csv(dest, index=False)
    logger.info(f"✓ BPE téléchargé : {len(df_dep):,} équipements (dép. {settings.department_code}) → {dest.name}")


def download_all() -> None:
    """Télécharger toutes les sources de données."""
    download_dvf_geolocalisees()
    download_dpe()
    download_bpe()


if __name__ == "__main__":
    download_all()
