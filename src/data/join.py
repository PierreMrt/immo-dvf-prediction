"""
Jointures entre DVF et sources externes : DPE, OSM (équipements), IRIS.
"""

import pandas as pd
import geopandas as gpd

from src.data.load import load_dpe, load_dvf_clean, load_equipements_osm
from src.utils.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.utils.logging import logger

RAYON_EQUIPEMENTS_M = 500
EPSG_WGS84 = 4326
EPSG_LAMBERT = 2154


def join_iris(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigner le code IRIS à chaque vente DVF via jointure géospatiale.
    Nécessite le fichier contours_iris_49.geojson dans data/raw/iris/.
    """
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
    Joindre les données DPE au dataset DVF par adresse normalisée.

    Colonnes apportées depuis dpe03existant :
      - etiquette_dpe, etiquette_ges
      - annee_construction
      - conso_5_usages_par_m2_ep
      - emission_ges_5_usages_par_m2
    """
    logger.info("Jointure DPE...")
    dpe = load_dpe()

    dpe["adresse_norm"] = (
        dpe["adresse_ban"]
        .str.upper().str.strip()
        .str.replace(r"[^A-Z0-9 ]", "", regex=True)
    )

    df = df.copy()
    # Colonnes d'adresse DVF brutes (non renommées par clean.py)
    num_col = next((c for c in ["adresse_numero", "numero_voie"] if c in df.columns), None)
    voie_col = next((c for c in ["adresse_nom_voie", "nom_voie"] if c in df.columns), None)

    if num_col and voie_col:
        df["adresse_norm"] = (
            (df[num_col].fillna("").astype(str) + " " + df[voie_col].fillna(""))
            .str.upper().str.strip()
            .str.replace(r"[^A-Z0-9 ]", "", regex=True)
        )
    else:
        logger.warning("⚠ Colonnes d'adresse DVF introuvables, jointure DPE ignorée")
        return df

    dpe_cols = [
        "adresse_norm",
        "etiquette_dpe",
        "etiquette_ges",
        "annee_construction",
        "conso_5_usages_par_m2_ep",
        "emission_ges_5_usages_par_m2",
    ]
    dpe_cols_available = [c for c in dpe_cols if c in dpe.columns]

    merged = df.merge(
        dpe[dpe_cols_available].drop_duplicates("adresse_norm"),
        on="adresse_norm",
        how="left",
    )

    n_matched = merged["etiquette_dpe"].notna().sum() if "etiquette_dpe" in merged.columns else 0
    logger.info(f"✓ DPE jointé : {n_matched:,} / {len(df):,} ({n_matched / len(df):.1%})")
    return merged.drop(columns=["adresse_norm"])


def join_equipements_osm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compter les équipements OSM dans un rayon de 500m autour de chaque vente DVF.
    """
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
    """Exécuter toutes les jointures dans l'ordre."""
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
