"""
Téléchargement des données DVF, DPE, équipements OSM, contours IRIS,
arrêts de transport (tram/gare) et zones inondables PPRI.

Sources :
- DVF géolocalisées : https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/{dep}.csv.gz
  Disponibles avec un décalage : l'année N est publiée courant N+1.
- DPE (depuis juil. 2021) : https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant
- Équipements OSM : Overpass API kumi.systems (une requête par filtre + retry)
- Contours IRIS : IGN via data.gouv.fr (GeoJSON du département)
- Arrêts tram/gare : data.angers.fr (tram) + OSM (gare SNCF)
- Zones inondables PPRI : data.gouv.fr (GeoJSON Maine-et-Loire)
"""

import gzip
import json
import shutil
import time
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import DATA_RAW_DIR, DVF_YEARS, settings
from src.utils.logging import logger

DVF_BASE_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/{dep}.csv.gz"
DPE_API_URL = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines"
DPE_BATCH_SIZE = 10_000
OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"

IRIS_BASE_URL = (
    "https://wxs.ign.fr/1yhlj2ehpqf3q6dt6a2y7b64/telechargement/inspire/"
    "CONTOURS-IRIS-2023-01-01$CONTOURS-IRIS_3-0__SHP__FRA_2023-01-01/"
    "file/CONTOURS-IRIS_3-0__SHP__FRA_2023-01-01.7z"
)
IRIS_GEOJSON_URL = (
    "https://data.geopf.fr/wfs/ows"
    "?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
    "&TYPENAMES=BDTOPO_V3:zone_d_activite_ou_d_interet"
    "&outputFormat=application/json"
)
IRIS_API_URL = (
    "https://data.geopf.fr/wfs/ows"
    "?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
    "&TYPENAMES=ADMINEXPRESS-COG-CARTO.LATEST:iris_ge"
    "&outputFormat=application/json"
    "&CQL_FILTER=insee_dep='{dep}'"
    "&count=5000"
)

# Arrêts tram Angers via data.angers.fr (API ODS)
TRAM_STOPS_URL = (
    "https://data.angers.fr/api/explore/v2.1/catalog/datasets/"
    "am_arrets_lignes/records?limit=100&where=mode_transport%3D'TRAM'"
)
# Gare Saint-Serge via Overpass (nœud OSM fixe)
GARE_OVERPASS_QUERY = (
    "[out:json][timeout:30];\n"
    "node[\"railway\"=\"station\"](47.40,-0.65,47.55,-0.45);\n"
    "out;"
)
# PPRI Maine-et-Loire — zones inondables data.gouv.fr
PPRI_URL = (
    "https://data.geopf.fr/wfs/ows"
    "?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
    "&TYPENAMES=BDTOPO_V3:zone_de_risque_naturel"
    "&outputFormat=application/json"
    "&CQL_FILTER=code_departement='49'"
    "&count=2000"
)

# Bbox zone Angers (~20km autour) : (lat_min, lon_min, lat_max, lon_max)
BBOX_ANGERS = (47.40, -0.65, 47.55, -0.45)

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

OSM_FILTERS: list[tuple[str, str]] = [
    ("commerce",   'node["shop"]'),
    ("restaurant", 'node["amenity"="restaurant"]'),
    ("restaurant", 'node["amenity"="fast_food"]'),
    ("ecole",      'node["amenity"="school"]'),
    ("ecole",      'node["amenity"="kindergarten"]'),
    ("parc",       'node["leisure"="park"]'),
    ("parc",       'node["leisure"="garden"]'),
]


def _download_stream(url: str, dest: Path, timeout: int = 600) -> None:
    """Télécharger un fichier en streaming vers dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65_536):
                f.write(chunk)


def _overpass_query_with_retry(
    osm_filter: str,
    bbox_str: str,
    overpass_timeout: int = 45,
    max_retries: int = 3,
    retry_delay: float = 15.0,
) -> list[dict]:
    """
    Exécuter une requête Overpass pour un filtre unique avec retry exponentiel.

    Args:
        osm_filter: Filtre Overpass sans bbox, ex : 'node["shop"]'
        bbox_str: Bbox au format 'lat_min,lon_min,lat_max,lon_max'
        overpass_timeout: Timeout Overpass QL (dans la requête)
        max_retries: Nombre de tentatives
        retry_delay: Délai initial entre tentatives (doublé à chaque échec)
    """
    query = (
        f"[out:json][timeout:{overpass_timeout}];\n"
        f"(\n  {osm_filter}({bbox_str});\n);\n"
        "out center;"
    )
    delay = retry_delay
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(
                OVERPASS_URL,
                params={"data": query},
                timeout=overpass_timeout + 10,
            )
            r.raise_for_status()
            return r.json().get("elements", [])
        except (requests.Timeout, requests.HTTPError) as e:
            if attempt < max_retries:
                logger.warning(f"    Tentative {attempt}/{max_retries} échouée ({e}), nouvel essai dans {delay:.0f}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise


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
    Une requête atomique par filtre OSM avec retry exponentiel.

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

    for categorie, osm_filter in OSM_FILTERS:
        logger.info(f"  {osm_filter}...")
        try:
            elements = _overpass_query_with_retry(osm_filter, bbox_str)
        except requests.RequestException as e:
            logger.error(f"✗ Erreur Overpass ({osm_filter}) après retries : {e}")
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
        time.sleep(2)

    if not all_rows:
        logger.warning("⚠ Aucun équipement OSM récupéré")
        return

    df = pd.DataFrame(all_rows).dropna(subset=["lat", "lon"])
    df.to_parquet(dest, index=False)
    logger.info(f"✓ Équipements OSM téléchargés : {len(df):,} éléments → {dest.name}")
    for cat in ["commerce", "restaurant", "ecole", "parc"]:
        logger.info(f"   {cat} : {(df['categorie'] == cat).sum():,}")


def download_iris(dep: str | None = None) -> None:
    """
    Télécharger les contours IRIS du département via l'API WFS Géoplateforme IGN.
    Sauvegarde : data/raw/iris/contours_iris_{dep}.geojson

    L'API retourne un GeoJSON paginé (paramètre `startIndex`).
    Si l'API échoue, un WARNING est émis sans bloquer le pipeline.
    """
    dep = dep or settings.department_code
    dest = DATA_RAW_DIR / "iris" / f"contours_iris_{dep}.geojson"

    if dest.exists():
        logger.info(f"Contours IRIS déjà présents, saut ({dest.name})")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Téléchargement contours IRIS (département {dep})...")

    base_url = (
        "https://data.geopf.fr/wfs/ows"
        "?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
        "&TYPENAMES=ADMINEXPRESS-COG-CARTO.LATEST:iris_ge"
        "&outputFormat=application/json"
        f"&CQL_FILTER=insee_dep='{dep}'"
        "&count=1000"
    )

    all_features: list[dict] = []
    start_index = 0

    while True:
        url = f"{base_url}&startIndex={start_index}"
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            logger.warning(f"⚠ Impossible de télécharger les contours IRIS ({e}), code_iris non assigné")
            return

        features = data.get("features", [])
        if not features:
            break

        all_features.extend(features)
        start_index += len(features)

        if len(features) < 1000:
            break

    if not all_features:
        logger.warning(f"⚠ Aucun contour IRIS reçu pour le département {dep}, code_iris non assigné")
        return

    geojson = {
        "type": "FeatureCollection",
        "features": all_features,
    }

    dest.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
    logger.info(f"✓ Contours IRIS téléchargés : {len(all_features)} zones → {dest.name}")


def download_arrets_transport() -> None:
    """
    Télécharger les arrêts tram depuis data.angers.fr et la gare SNCF via Overpass.
    Sauvegarde : data/raw/transport/arrets_transport.parquet

    - Tram : API ODS data.angers.fr, filtre mode_transport=TRAM
    - Gare : nœud OSM railway=station dans la bbox Angers
    """
    dest = DATA_RAW_DIR / "transport" / "arrets_transport.parquet"
    if dest.exists():
        logger.info(f"Arrêts transport déjà présents, saut ({dest.name})")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # --- Arrêts tram via data.angers.fr ---
    logger.info("Téléchargement arrêts tram (data.angers.fr)...")
    offset = 0
    limit = 100
    while True:
        try:
            r = requests.get(
                "https://data.angers.fr/api/explore/v2.1/catalog/datasets/"
                "am_arrets_lignes/records",
                params={"limit": limit, "offset": offset, "where": "mode_transport='TRAM'"},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            logger.warning(f"⚠ Erreur arrêts tram ({e})")
            break

        records = data.get("results", [])
        if not records:
            break

        for rec in records:
            geo = rec.get("geo_point_2d") or {}
            lat = geo.get("lat") or rec.get("lat")
            lon = geo.get("lon") or rec.get("lon")
            if lat and lon:
                rows.append({
                    "lat": float(lat),
                    "lon": float(lon),
                    "type": "tram",
                    "nom": rec.get("nom_arret", ""),
                })

        offset += limit
        if len(records) < limit:
            break

    logger.info(f"  ✓ {sum(1 for r in rows if r['type'] == 'tram')} arrêts tram")

    # --- Gare SNCF via Overpass ---
    logger.info("Téléchargement gares SNCF (Overpass)...")
    try:
        r = requests.get(
            OVERPASS_URL,
            params={"data": GARE_OVERPASS_QUERY},
            timeout=40,
        )
        r.raise_for_status()
        elements = r.json().get("elements", [])
        for el in elements:
            tags = el.get("tags", {})
            rows.append({
                "lat": el["lat"],
                "lon": el["lon"],
                "type": "gare",
                "nom": tags.get("name", ""),
            })
        logger.info(f"  ✓ {len(elements)} gare(s) SNCF")
    except requests.RequestException as e:
        logger.warning(f"⚠ Erreur gares Overpass ({e})")

    if not rows:
        logger.warning("⚠ Aucun arrêt transport récupéré")
        return

    df = pd.DataFrame(rows)
    df.to_parquet(dest, index=False)
    logger.info(f"✓ Arrêts transport sauvegardés : {len(df)} → {dest.name}")


def download_ppri() -> None:
    """
    Télécharger les zones inondables PPRI du Maine-et-Loire via la Géoplateforme IGN.
    Sauvegarde : data/raw/ppri/zones_inondables_49.geojson

    Utilise le WFS BDTOPO zone_de_risque_naturel filtré sur le département 49.
    Si l'API échoue, un WARNING est émis sans bloquer le pipeline.
    """
    dest = DATA_RAW_DIR / "ppri" / "zones_inondables_49.geojson"
    if dest.exists():
        logger.info(f"PPRI déjà présent, saut ({dest.name})")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Téléchargement zones inondables PPRI (département 49)...")

    all_features: list[dict] = []
    start_index = 0

    while True:
        url = f"{PPRI_URL}&startIndex={start_index}"
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            logger.warning(f"⚠ Impossible de télécharger le PPRI ({e}), zone_inondable non assignée")
            return

        features = data.get("features", [])
        if not features:
            break

        all_features.extend(features)
        start_index += len(features)

        if len(features) < 2000:
            break

    if not all_features:
        logger.warning("⚠ Aucune zone PPRI reçue, zone_inondable non assignée")
        return

    geojson = {"type": "FeatureCollection", "features": all_features}
    dest.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
    logger.info(f"✓ PPRI téléchargé : {len(all_features)} zones → {dest.name}")


def download_all() -> None:
    """Télécharger toutes les sources de données."""
    download_dvf_geolocalisees()
    download_dpe()
    download_equipements_osm()
    download_iris()
    download_arrets_transport()
    download_ppri()


if __name__ == "__main__":
    download_all()
