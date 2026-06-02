"""
Téléchargement des données DVF, DPE et équipements OSM depuis leurs URLs stables.

Sources :
- DVF géolocalisées : https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/{dep}.csv.gz
  Disponibles avec un décalage : l'année N est publiée courant N+1.
- DPE (depuis juil. 2021) : https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant
- Équipements OSM : Overpass API kumi.systems (requêtes par catégorie)
"""

import gzip
import shutil
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import DATA_RAW_DIR, DVF_YEARS, settings
from src.utils.logging import logger

DVF_BASE_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/{dep}.csv.gz"
DPE_API_URL = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines"
DPE_BATCH_SIZE = 10_000
OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"

# Bbox restreinte à Angers et communes limitrophes (~20km autour)
# Plus rapide qu'une bbox départementale, couvre la zone d'intérêt du projet
BBOX_ANGERS = (47.40, -0.65, 47.55, -0.45)

# Colonnes à conserver depuis dpe03existant (noms confirmés via API)
DPE_COLS_TO_KEEP = [
    "numero_dpe",
    "date_etablissement_dpe",
    "adresse_ban",
    "code_postal_ban",
    "nom_commune_ban",
    "etiquette_dpe",
    "etiquette_ges",
    "annee_construction",
    "surface_habitable_immeuble",
    "conso_5_usages_par_m2_ep",
    "emission_ges_5_usages_par_m2",
    "coordonnee_cartographique_x_ban",
    "coordonnee_cartographique_y_ban",
    "periode_construction",
    "type_batiment",
]

# Catégories OSM : chaque entrée est (categorie_sortie, filtres_overpass)
OSM_CATEGORIES: list[tuple[str, list[str]]] = [
    ("commerce",    ['node["shop"]']),
    ("restaurant",  ['node["amenity"="restaurant"]', 'node["amenity"="fast_food"]']),
    ("ecole",       ['node["amenity"="school"]', 'node["amenity"="kindergarten"]']),
    ("parc",        ['node["leisure"="park"]', 'node["leisure"="garden"]']),
]


def _download_stream(url: str, dest: Path, timeout: int = 600) -> None:
    """Télécharger un fichier en streaming vers dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65_536):
                f.write(chunk)


def _overpass_query(filters: list[str], bbox_str: str, timeout: int = 60) -> list[dict]:
    """
    Exécuter une requête Overpass pour une liste de filtres sur une bbox.
    Retourne la liste des éléments OSM.
    """
    lines = "\n".join(f"  {f}({bbox_str});" for f in filters)
    query = f"[out:json][timeout:{timeout}];\n(\n{lines}\n);\nout center;"
    r = requests.get(OVERPASS_URL, params={"data": query}, timeout=timeout + 10)
    r.raise_for_status()
    return r.json().get("elements", [])


def download_dvf_geolocalisees(years: list[int] | None = None) -> None:
    """
    Télécharger les DVF géolocalisées pour le département configuré.
    Les fichiers .csv.gz sont décompressés automatiquement en .csv.
    L'année N est généralement disponible courant N+1.
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
                logger.warning(f"⚠ DVF {year} non disponible (publiée avec décalage, réessayer plus tard)")
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
    Filtre sur le préfixe de code postal via code_postal_ban_starts.
    La pagination suit directement l'URL 'next' retournée par l'API.
    """
    dest = DATA_RAW_DIR / "dpe" / "dpe_angers.parquet"
    if dest.exists():
        logger.info(f"DPE déjà présent, saut ({dest.name})")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Téléchargement DPE ADEME (code postal {code_postal_prefix}*)...")

    next_url: str | None = (
        f"{DPE_API_URL}?size={DPE_BATCH_SIZE}&code_postal_ban_starts={code_postal_prefix}"
    )

    dfs: list[pd.DataFrame] = []
    page = 1

    while next_url:
        try:
            r = requests.get(next_url, timeout=60)
            r.raise_for_status()
            data = r.json()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            logger.error(f"✗ Erreur DPE page {page} ({status}) : {e}")
            break
        except requests.RequestException as e:
            logger.error(f"✗ Erreur DPE page {page} : {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        df_page = pd.DataFrame(results)
        cols = [c for c in DPE_COLS_TO_KEEP if c in df_page.columns]
        dfs.append(df_page[cols])
        logger.info(f"  Page {page} : {len(results)} enregistrements")

        next_url = data.get("next")
        page += 1

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df.to_parquet(dest, index=False)
        logger.info(f"✓ DPE téléchargé : {len(df):,} lignes → {dest.name}")
    else:
        logger.warning("⚠ Aucune donnée DPE récupérée")


def download_equipements_osm(bbox: tuple[float, float, float, float] | None = None) -> None:
    """
    Télécharger les équipements urbains via Overpass API (instance kumi.systems).
    Une requête par catégorie pour éviter les timeouts.

    Args:
        bbox: (lat_min, lon_min, lat_max, lon_max) — défaut : zone Angers
    """
    dest = DATA_RAW_DIR / "osm" / "equipements_osm.parquet"
    if dest.exists():
        logger.info(f"Equipements OSM déjà présents, saut ({dest.name})")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    bbox = bbox or BBOX_ANGERS
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    logger.info(f"Téléchargement équipements OSM (bbox {bbox_str})...")

    all_rows: list[dict] = []

    for categorie, filters in OSM_CATEGORIES:
        logger.info(f"  Requête OSM : {categorie}...")
        try:
            elements = _overpass_query(filters, bbox_str)
        except requests.RequestException as e:
            logger.error(f"✗ Erreur Overpass ({categorie}) : {e}")
            continue

        for el in elements:
            tags = el.get("tags", {})
            all_rows.append({
                "osm_id": el["id"],
                "lat": el.get("lat") or el.get("center", {}).get("lat"),
                "lon": el.get("lon") or el.get("center", {}).get("lon"),
                "categorie": categorie,
                "nom": tags.get("name", ""),
            })
        logger.info(f"    ✓ {len(elements)} éléments")

    if not all_rows:
        logger.warning("⚠ Aucun équipement OSM récupéré")
        return

    df = pd.DataFrame(all_rows).dropna(subset=["lat", "lon"])
    df.to_parquet(dest, index=False)
    logger.info(f"✓ Équipements OSM téléchargés : {len(df):,} éléments → {dest.name}")


def download_all() -> None:
    """Télécharger toutes les sources de données."""
    download_dvf_geolocalisees()
    download_dpe()
    download_equipements_osm()


if __name__ == "__main__":
    download_all()
