"""
Jointures entre DVF et sources externes : DPE, OSM (équipements), IRIS.
"""

import geopandas as gpd
import pandas as pd

from src.data.load import load_dpe, load_dvf_clean, load_equipements_osm
from src.utils.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.utils.logging import logger

RAYON_EQUIPEMENTS_M = 500
DPE_SEUIL_M = 50
EPSG_WGS84 = 4326
EPSG_LAMBERT = 2154


def join_iris(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Jointure IRIS...")
    iris_path = DATA_RAW_DIR / "iris" / "contours_iris_49.geojson"
    if not iris_path.exists():
        logger.warning(f"Contours IRIS introuvables ({iris_path.name}), code_iris non assigné")
        return df

    iris_geo = gpd.read_file(iris_path)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=EPSG_WGS84,
    )
    joined = gpd.sjoin(
        gdf, iris_geo[["CODE_IRIS", "NOM_IRIS", "geometry"]], how="left", predicate="within"
    )
    df = df.copy()
    df["code_iris"] = joined["CODE_IRIS"].values
    df["nom_iris"] = joined["NOM_IRIS"].values
    logger.info(f"✓ IRIS assigné ({df['code_iris'].notna().sum():,} / {len(df):,})")
    return df


def join_dpe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Joindre les données DPE au dataset DVF par nearest neighbor géospatial.
    Distance max : DPE_SEUIL_M mètres (Lambert-93).
    """
    logger.info(f"Jointure DPE (nearest neighbor, seuil {DPE_SEUIL_M}m)...")
    dpe = load_dpe()

    x_col, y_col = "coordonnee_cartographique_x_ban", "coordonnee_cartographique_y_ban"
    if x_col not in dpe.columns or y_col not in dpe.columns:
        logger.warning("⚠ Coordonnées DPE manquantes, jointure ignorée")
        return df

    dpe_valid = dpe.dropna(subset=[x_col, y_col]).copy()
    logger.info(f"  DPE avec coordonnées : {len(dpe_valid):,} / {len(dpe):,}")

    dpe_cols_keep = [
        "etiquette_dpe", "etiquette_ges", "annee_construction",
        "conso_5_usages_par_m2_ep", "emission_ges_5_usages_par_m2",
    ]
    dpe_cols_available = [c for c in dpe_cols_keep if c in dpe_valid.columns]

    gdf_dpe = gpd.GeoDataFrame(
        dpe_valid[dpe_cols_available],
        geometry=gpd.points_from_xy(dpe_valid[x_col], dpe_valid[y_col]),
        crs=EPSG_LAMBERT,
    ).reset_index(drop=True)

    # Conserver l'index DVF original pour la déduplication
    df = df.reset_index(drop=True).copy()
    df["_dvf_idx"] = df.index

    gdf_dvf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=EPSG_WGS84,
    ).to_crs(EPSG_LAMBERT)

    joined = gpd.sjoin_nearest(
        gdf_dvf,
        gdf_dpe,
        how="left",
        max_distance=DPE_SEUIL_M,
        distance_col="_dpe_dist_m",
    )

    # En cas de doublons (plusieurs DPE à égalité), garder le plus proche
    joined = (
        joined
        .sort_values("_dpe_dist_m")
        .drop_duplicates(subset="_dvf_idx", keep="first")
        .set_index("_dvf_idx")
        .sort_index()
    )

    result = df.drop(columns=["_dvf_idx"]).copy()
    for col in dpe_cols_available:
        result[col] = joined[col].values

    n_matched = result["etiquette_dpe"].notna().sum() if "etiquette_dpe" in result.columns else 0
    logger.info(f"✓ DPE jointé : {n_matched:,} / {len(df):,} ({n_matched / len(df):.1%})")
    return result


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


def run_all_joins(df: pd.DataFrame) -> pd.DataFrame:
    df = join_iris(df)
    df = join_dpe(df)
    df = join_equipements_osm(df)
    return df


if __name__ == "__main__":
    df = load_dvf_clean()
    df = run_all_joins(df)
    output = DATA_PROCESSED_DIR / "dvf_angers_joined.parquet"
    df.to_parquet(output, index=False)
    logger.info(f"✓ Dataset enrichi sauvegardé : {output.name}")
