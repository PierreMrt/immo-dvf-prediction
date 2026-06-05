"""
Téléchargement des données DVF, DPE, équipements OSM, contours IRIS,
arrêts de transport (tram/gare) et zones inondables PPRI.

Sources :
- DVF géolisées : https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/{dep}.csv.gz
  Disponibles avec un décalage : l'année N est publiée courant N+1.
- DPE (depuis juil. 2021) : https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant
- Équipements OSM : Overpass API kumi.systems (une requête par filtre + retry)
- Contours IRIS : IGN via WFS data.geopf.fr (STATISTICALUNITS.IRISGE:iris_ge)
- Arrêts tram/gare : data.angers.fr (GTFS stops Irigo) + OSM (gare SNCF)
- Zones inondables PPRI : Géorisques /api/v1/gaspar/azi (rayon + latlon, max ~20 km)
  Retourne une entrée par couple (AZI, commune) avec code_insee directement dans le record.
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

# Instances Overpass par ordre de préférence — fallback si timeout/erreur
# Les deux acceptent les requêtes POST form-encoded : data=<query>
OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

# Contours IRIS via WFS data.geopf.fr (endpoint public, sans clé)
IRIS_WFS_URL = (
    "https://data.geopf.fr/wfs/ows"
    "?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
    "&TYPENAMES=STATISTICALUNITS.IRISGE:iris_ge"
    "&outputFormat=application/json"
    "&CQL_FILTER=code_insee+LIKE+'{dep}%25'"
    "&count=1000"
)

# Arrêts tram Angers via data.angers.fr (dataset GTFS stops Irigo)
TRAM_DATASET_ID = "horaires-theoriques-et-arrets-du-reseau-irigo-gtfs"

# Gare SNCF via Overpass — bbox élargie pour couvrir Angers Saint-Laud (lon ~-0.5526)
# Format Overpass : lat_min,lon_min,lat_max,lon_max
GARE_OVERPASS_QUERY = (
    "[out:json][timeout:30];\n"
    "node[\"railway\"=\"station\"](47.40,-0.65,47.55,-0.50);\n"
    "out;\n"
)

# PPRI — Atlas des Zones Inondables via Géorisques /api/v1/gaspar/azi
# Retourne une entrée par couple (AZI, commune) avec code_insee directement dans le record.
PPRI_GASPAR_URL = "https://www.georisques.gouv.fr/api/v1/gaspar/azi"
PPRI_CENTER_LATLON = "-0.556,47.478"  # lon,lat centre Angers
PPRI_RAYON = 20_000  # 20 km (limite empirique de l'API)

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


def _overpass_post(url: str, query: str, timeout: int) -> requests.Response:
    """
    Envoyer une requête Overpass en POST form-encoded.
    Les deux instances (kumi.systems et overpass-api.de) exigent ce format.
    GET avec params= encode la query dans l'URL → 406 sur overpass-api.de.
    """
    return requests.post(
        url,
        data={"data": query},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout,
    )


def _overpass_query_with_retry(
    osm_filter: str,
    bbox_str: str,
    overpass_timeout: int = 45,
    max_retries: int = 3,
    retry_delay: float = 15.0,
) -> list[dict]:
    """
    Exécuter une requête Overpass pour un filtre unique avec retry exponentiel.
    Essaie les instances OVERPASS_URLS dans l'ordre en cas de timeout/erreur.

    Args:
        osm_filter: Filtre Overpass sans bbox, ex : 'node["shop"]'
        bbox_str: Bbox au format 'lat_min,lon_min,lat_max,lon_max'
        overpass_timeout: Timeout Overpass QL (dans la requête)
        max_retries: Nombre de tentatives par instance
        retry_delay: Délai initial entre tentatives (doublé à chaque échec)
    """
    query = (
        f"[out:json][timeout:{overpass_timeout}];\n"
        f"(\n  {osm_filter}({bbox_str});\n);\n"
        "out center;"
    )
    for overpass_url in OVERPASS_URLS:
        delay = retry_delay
        for attempt in range(1, max_retries + 1):
            try:
                r = _overpass_post(overpass_url, query, timeout=overpass_timeout + 10)
                r.raise_for_status()
                return r.json().get("elements", [])
            except (requests.Timeout, requests.HTTPError) as e:
                if attempt < max_retries:
                    logger.warning(
                        f"    [{overpass_url}] Tentative {attempt}/{max_retries} échouée ({e}), "
                        f"nouvel essai dans {delay:.0f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.warning(f"    [{overpass_url}] échoué après {max_retries} tentatives, essai instance suivante...")
    raise requests.RequestException(f"Toutes les instances Overpass ont échoué pour : {osm_filter}")


def _overpass_raw_with_fallback(query: str, timeout: int = 40) -> list[dict]:
    """
    Exécuter une requête Overpass brute (query complète) en POST avec fallback d'instance.
    Utilisé pour les requêtes ponctuelles (ex : gares SNCF).
    POST form-encoded requis — GET provoque un 406 sur overpass-api.de.
    """
    for overpass_url in OVERPASS_URLS:
        try:
            r = _overpass_post(overpass_url, query, timeout=timeout)
            r.raise_for_status()
            return r.json().get("elements", [])
        except requests.RequestException as e:
            logger.warning(f"    [{overpass_url}] échoué ({e}), essai instance suivante...")
    raise requests.RequestException("Toutes les instances Overpass ont échoué")


def download_dvf_geolocalisees(years: list[int] | None = None) -> None:
    """
    Télécharger les DVF géolisées pour le département configuré.
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
    Télécharger les équipements urbains via Overpass API.
    Une requête atomique par filtre OSM avec retry exponentiel et fallback d'instance.

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
    Télécharger les contours IRIS du département via le WFS data.geopf.fr.
    Sauvegarde : data/raw/iris/contours_iris_{dep}.geojson

    Typename WFS : STATISTICALUNITS.IRISGE:iris_ge
    Endpoint : data.geopf.fr/wfs/ows (public, sans clé API)
    Filtre : code_insee LIKE '{dep}%'
    L'API retourne un GeoJSON paginé (paramètre startIndex).
    Si l'API échoue, un WARNING est émis sans bloquer le pipeline.
    """
    dep = dep or settings.department_code
    dest = DATA_RAW_DIR / "iris" / f"contours_iris_{dep}.geojson"

    if dest.exists():
        logger.info(f"Contours IRIS déjà présents, saut ({dest.name})")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Téléchargement contours IRIS (département {dep})...")

    base_url = IRIS_WFS_URL.format(dep=dep)
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

    - Tram : dataset GTFS stops Irigo (tous les stops, sans filtre route_type absent du dataset)
    - Gare : nœud OSM railway=station dans la bbox élargie couvrant Saint-Laud (lon ~-0.5526)
      Envoi en POST form-encoded — fallback automatique sur overpass-api.de si kumi.systems timeout.
    """
    dest = DATA_RAW_DIR / "transport" / "arrets_transport.parquet"
    if dest.exists():
        logger.info(f"Arrêts transport déjà présents, saut ({dest.name})")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # --- Arrêts tram via data.angers.fr (dataset GTFS stops) ---
    logger.info("Téléchargement arrêts tram (data.angers.fr - GTFS Irigo)...")
    offset = 0
    limit = 100
    tram_count = 0
    while True:
        try:
            r = requests.get(
                f"https://data.angers.fr/api/explore/v2.1/catalog/datasets/{TRAM_DATASET_ID}/records",
                params={"limit": limit, "offset": offset},
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
            geo = rec.get("stop_coordinates") or rec.get("geo_point_2d") or {}
            lat = geo.get("lat") or rec.get("stop_lat")
            lon = geo.get("lon") or rec.get("stop_lon")
            if lat and lon:
                rows.append({
                    "lat": float(lat),
                    "lon": float(lon),
                    "type": "tram",
                    "nom": rec.get("stop_name", ""),
                })
                tram_count += 1

        offset += limit
        if len(records) < limit:
            break

    logger.info(f"  ✓ {tram_count} arrêts tram")

    # --- Gare SNCF via Overpass POST (avec fallback d'instance) ---
    logger.info("Téléchargement gares SNCF (Overpass)...")
    try:
        elements = _overpass_raw_with_fallback(GARE_OVERPASS_QUERY, timeout=40)
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
        logger.warning(f"⚠ Erreur gares Overpass ({e}), distance_gare_proche_m sera NaN")

    if not rows:
        logger.warning("⚠ Aucun arrêt transport récupéré")
        return

    df = pd.DataFrame(rows)
    df.to_parquet(dest, index=False)
    logger.info(f"✓ Arrêts transport sauvegardés : {len(df)} → {dest.name}")


def download_ppri() -> None:
    """
    Télécharger les zones inondables PPRI via Géorisques /api/v1/gaspar/azi.
    Sauvegarde : data/raw/ppri/zones_inondables_49.json

    L'API retourne une entrée par couple (AZI, commune) avec code_insee directement
    dans chaque record (pas dans une liste imbriquée).
    Pagination complète jusqu'à épuisement des pages.
    Si l'API échoue, un WARNING est émis sans bloquer le pipeline.
    """
    dest = DATA_RAW_DIR / "ppri" / "zones_inondables_49.json"
    if dest.exists():
        logger.info(f"PPRI déjà présent, saut ({dest.name})")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Téléchargement zones inondables PPRI (Angers, rayon 20 km)...")

    all_records: list[dict] = []
    page = 1
    page_size = 100

    while True:
        try:
            r = requests.get(
                PPRI_GASPAR_URL,
                params={
                    "rayon": PPRI_RAYON,
                    "latlon": PPRI_CENTER_LATLON,
                    "page": page,
                    "page_size": page_size,
                },
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            logger.warning(f"⚠ Impossible de télécharger le PPRI ({e}), zone_inondable non assignée")
            return

        records = data.get("data", [])
        all_records.extend(records)

        total_pages = data.get("total_pages", 1)
        if page >= total_pages or len(records) < page_size:
            break
        page += 1

    if not all_records:
        logger.warning("⚠ Aucune zone PPRI reçue, zone_inondable non assignée")
        return

    dest.write_text(json.dumps(all_records, ensure_ascii=False), encoding="utf-8")
    logger.info(f"✓ PPRI téléchargé : {len(all_records)} entrées → {dest.name}")


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
