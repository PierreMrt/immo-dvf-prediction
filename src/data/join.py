"""
Jointures entre DVF et sources externes : DPE, OSM (équipements),
IRIS, transports (tram/gare), zones inondables PPRI.
"""

import numpy as np
import geopandas as gpd
import pandas as pd

from src.data.clean import normalize_adresse_ban
from src.data.load import load_dpe, load_dvf_clean, load_equipements_osm
from src.utils.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.utils.logging import logger

RAYON_EQUIPEMENTS_M = 500
EPSG_WGS84 = 4326
EPSG_LAMBERT = 2154


def _find_iris_columns(iris_geo: gpd.GeoDataFrame) -> tuple[str, str]:
    """
    Retrouver les colonnes code_iris et nom_iris quelle que soit leur casse.
    Le WFS data.geopf.fr peut retourner CODE_IRIS ou code_iris selon les versions.
    """
    cols_lower = {c.lower(): c for c in iris_geo.columns}
    col_code = cols_lower.get("code_iris")
    col_nom = cols_lower.get("nom_iris")
    if col_code is None or col_nom is None:
        raise KeyError(
            f"Colonnes code_iris / nom_iris introuvables dans le GeoJSON IRIS. "
            f"Colonnes disponibles : {list(iris_geo.columns)}"
        )
    return col_code, col_nom


def join_iris(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Jointure IRIS...")
    iris_path = DATA_RAW_DIR / "iris" / "contours_iris_49.geojson"
    if not iris_path.exists():
        logger.warning(f"Contours IRIS introuvables ({iris_path.name}), code_iris non assigné")
        return df

    iris_geo = gpd.read_file(iris_path)
    col_code, col_nom = _find_iris_columns(iris_geo)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=EPSG_WGS84,
    )
    joined = gpd.sjoin(
        gdf, iris_geo[[col_code, col_nom, "geometry"]], how="left", predicate="within"
    )
    df = df.copy()
    df["code_iris"] = joined[col_code].values
    df["nom_iris"] = joined[col_nom].values
    logger.info(f"✓ IRIS assigné ({df['code_iris'].notna().sum():,} / {len(df):,})")
    return df


def join_dpe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Joindre les données DPE au dataset DVF par adresse normalisée.

    Prérequis : df doit contenir la colonne 'adresse_norm' (produite par clean.py).

    Stratégie :
    - Normaliser adresse_ban DPE via normalize_adresse_ban() (clean.py)
    - Jointure sur la clé adresse_norm
    - En cas de doublons DPE sur la même adresse, prendre le DPE le plus récent
    """
    logger.info("Jointure DPE (par adresse normalisée)...")

    if "adresse_norm" not in df.columns:
        logger.warning("⚠ Colonne adresse_norm absente — relancer make data-clean d'abord")
        return df

    dpe = load_dpe()

    dpe_cols_keep = [
        "adresse_ban",
        "etiquette_dpe",
        "etiquette_ges",
        "annee_construction",
        "conso_5_usages_par_m2_ep",
        "emission_ges_5_usages_par_m2",
        "date_etablissement_dpe",
    ]
    dpe_cols_available = [c for c in dpe_cols_keep if c in dpe.columns]
    dpe = dpe[dpe_cols_available].copy()

    dpe["adresse_norm"] = normalize_adresse_ban(dpe["adresse_ban"])

    if "date_etablissement_dpe" in dpe.columns:
        dpe = (
            dpe.sort_values("date_etablissement_dpe", ascending=False)
            .drop_duplicates(subset="adresse_norm", keep="first")
        )
    else:
        dpe = dpe.drop_duplicates(subset="adresse_norm", keep="first")

    dpe = dpe.drop(columns=["adresse_ban", "date_etablissement_dpe"], errors="ignore")

    logger.info(f"  Exemple clés DPE  : {dpe['adresse_norm'].head(5).tolist()}")
    logger.info(f"  Exemple clés DVF  : {df['adresse_norm'].head(5).tolist()}")

    merged = df.merge(dpe, on="adresse_norm", how="left")

    n_matched = merged["etiquette_dpe"].notna().sum() if "etiquette_dpe" in merged.columns else 0
    logger.info(f"✓ DPE jointé : {n_matched:,} / {len(df):,} ({n_matched / len(df):.1%})")
    return merged


def join_equipements_osm(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Jointure équipements OSM (rayon {RAYON_EQUIPEMENTS_M}m)...")
    osm = load_equipements_osm()

    gdf_dvf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=EPSG_WGS84,
    ).to_crs(EPSG_LAMBERT)

    gdf_osm = gpd.GeoDataFrame(
        osm,
        geometry=gpd.points_from_xy(osm["lon"], osm["lat"]),
        crs=EPSG_WGS84,
    ).to_crs(EPSG_LAMBERT)

    categories = {
        "nb_commerces_500m": "commerce",
        "nb_restaurants_500m": "restaurant",
        "nb_ecoles_500m": "ecole",
        "nb_parcs_500m": "parc",
    }

    df = df.copy()
    for col_name, categorie in categories.items():
        subset = gdf_osm[gdf_osm["categorie"] == categorie]
        df[col_name] = [
            subset[subset.geometry.within(geom.buffer(RAYON_EQUIPEMENTS_M))].shape[0]
            for geom in gdf_dvf.geometry
        ]

    logger.info("✓ Équipements OSM jointés")
    return df


def join_transports(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculer la distance au tram le plus proche et à la gare la plus proche.

    Prérequis : data/raw/transport/arrets_transport.parquet
    (produit par download_arrets_transport)

    Colonnes ajoutées :
    - distance_tram_proche_m : distance en mètres à l'arrêt de tram le plus proche
    - distance_gare_proche_m : distance en mètres à la gare SNCF la plus proche
    """
    transport_path = DATA_RAW_DIR / "transport" / "arrets_transport.parquet"
    if not transport_path.exists():
        logger.warning("⚠ Arrêts transport introuvables, distances non calculées")
        return df

    logger.info("Jointure transports (distance tram/gare)...")
    transport = pd.read_parquet(transport_path)

    gdf_dvf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=EPSG_WGS84,
    ).to_crs(EPSG_LAMBERT)

    df = df.copy()

    for transport_type, col_name in [("tram", "distance_tram_proche_m"), ("gare", "distance_gare_proche_m")]:
        subset = transport[transport["type"] == transport_type]
        if subset.empty:
            logger.warning(f"⚠ Aucun arrêt de type '{transport_type}' trouvé")
            df[col_name] = np.nan
            continue

        gdf_transport = gpd.GeoDataFrame(
            subset,
            geometry=gpd.points_from_xy(subset["lon"], subset["lat"]),
            crs=EPSG_WGS84,
        ).to_crs(EPSG_LAMBERT)

        distances = gdf_dvf.geometry.apply(
            lambda pt: gdf_transport.geometry.distance(pt).min()
        )
        df[col_name] = distances.values

    n_tram = df["distance_tram_proche_m"].notna().sum()
    logger.info(f"✓ Distances transport calculées ({n_tram:,} biens)")
    return df


def join_ppri(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigner une zone inondable (booléen) à chaque bien via le PPRI.

    Prérequis : data/raw/ppri/zones_inondables_49.geojson
    (produit par download_ppri)

    Colonne ajoutée :
    - zone_inondable : 1 si le bien est dans une zone inondable PPRI, 0 sinon
    """
    ppri_path = DATA_RAW_DIR / "ppri" / "zones_inondables_49.geojson"
    if not ppri_path.exists():
        logger.warning("⚠ PPRI introuvable, zone_inondable non assignée")
        return df

    logger.info("Jointure PPRI (zones inondables)...")
    ppri = gpd.read_file(ppri_path).to_crs(EPSG_LAMBERT)

    gdf_dvf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=EPSG_WGS84,
    ).to_crs(EPSG_LAMBERT)

    joined = gpd.sjoin(gdf_dvf, ppri[["geometry"]], how="left", predicate="within")
    df = df.copy()
    df["zone_inondable"] = joined.index.isin(
        joined[joined["index_right"].notna()].index
    ).astype(int)

    n_inond = df["zone_inondable"].sum()
    logger.info(f"✓ Zone inondable assignée ({n_inond:,} biens en zone PPRI)")
    return df


def run_all_joins(df: pd.DataFrame) -> pd.DataFrame:
    df = join_iris(df)
    df = join_dpe(df)
    df = join_equipements_osm(df)
    df = join_transports(df)
    df = join_ppri(df)
    return df


if __name__ == "__main__":
    df = load_dvf_clean()
    df = run_all_joins(df)
    output = DATA_PROCESSED_DIR / "dvf_angers_joined.parquet"
    df.to_parquet(output, index=False)
    logger.info(f"✓ Dataset enrichi sauvegardé : {output.name}")
