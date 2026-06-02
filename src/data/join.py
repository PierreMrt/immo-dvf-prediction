"""
Jointures entre DVF et sources externes : DPE, BPE, IRIS.
"""

import pandas as pd
import geopandas as gpd

from src.data.load import load_bpe, load_dpe, load_dvf_clean
from src.utils.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.utils.logging import logger

RAYON_EQUIPEMENTS_M = 500
EPSG_WGS84 = 4326
EPSG_LAMBERT = 2154  # Lambert-93 (unité en mètres)


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
      - etiquette_dpe   : classe énergie (A-G)
      - etiquette_ges   : classe GES (A-G)
      - annee_construction
      - conso_5_usages_par_m2_ep
      - emission_ges_5_usages_par_m2
    """
    logger.info("Jointure DPE...")
    dpe = load_dpe()

    dpe["adresse_norm"] = (
        dpe["adresse_ban"]
        .str.upper()
        .str.strip()
        .str.replace(r"[^A-Z0-9 ]", "", regex=True)
    )
    df = df.copy()
    df["adresse_norm"] = (
        (df["numero_voie"].fillna("").astype(str) + " " + df["nom_voie"].fillna(""))
        .str.upper()
        .str.strip()
        .str.replace(r"[^A-Z0-9 ]", "", regex=True)
    )

    dpe_cols = [
        "adresse_norm",
        "etiquette_dpe",
        "etiquette_ges",
        "annee_construction",
        "conso_5_usages_par_m2_ep",
        "emission_ges_5_usages_par_m2",
    ]
    # Ne garder que les colonnes disponibles dans le fichier téléchargé
    dpe_cols_available = [c for c in dpe_cols if c in dpe.columns]

    merged = df.merge(
        dpe[dpe_cols_available].drop_duplicates("adresse_norm"),
        on="adresse_norm",
        how="left",
    )

    n_matched = merged["etiquette_dpe"].notna().sum() if "etiquette_dpe" in merged.columns else 0
    logger.info(f"✓ DPE jointé : {n_matched:,} / {len(df):,} ({n_matched / len(df):.1%})")
    return merged.drop(columns=["adresse_norm"])


def join_bpe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compter les équipements BPE dans un rayon de 500m autour de chaque vente.
    Utilise les coordonnées Lambert-93 de la BPE (colonnes LAMBERT_X / LAMBERT_Y).
    """
    logger.info(f"Jointure BPE (rayon {RAYON_EQUIPEMENTS_M}m)...")
    bpe = load_bpe()

    # Détection flexible des colonnes de coordonnées Lambert
    x_col = next((c for c in ["LAMBERT_X", "lambert_x", "X"] if c in bpe.columns), None)
    y_col = next((c for c in ["LAMBERT_Y", "lambert_y", "Y"] if c in bpe.columns), None)

    if not x_col or not y_col:
        logger.warning("Colonnes Lambert introuvables dans BPE, jointure ignorée")
        return df

    typequ_col = next((c for c in ["TYPEQU", "typequ"] if c in bpe.columns), None)
    if not typequ_col:
        logger.warning("Colonne TYPEQU introuvable dans BPE, jointure ignorée")
        return df

    gdf_dvf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=EPSG_WGS84,
    ).to_crs(EPSG_LAMBERT)

    gdf_bpe = gpd.GeoDataFrame(
        bpe,
        geometry=gpd.points_from_xy(bpe[x_col], bpe[y_col]),
        crs=EPSG_LAMBERT,
    )

    types_bpe = {
        "nb_commerces_500m": ["B101", "B102", "B201", "B202", "B203"],
        "nb_restaurants_500m": ["A504"],
        "nb_ecoles_500m": ["C101", "C102", "C201"],
        "nb_parcs_500m": ["F307"],
    }

    df = df.copy()
    for col_name, codes in types_bpe.items():
        bpe_subset = gdf_bpe[gdf_bpe[typequ_col].isin(codes)]
        df[col_name] = [
            bpe_subset[bpe_subset.geometry.within(geom.buffer(RAYON_EQUIPEMENTS_M))].shape[0]
            for geom in gdf_dvf.geometry
        ]

    logger.info("✓ BPE jointé")
    return df


def run_all_joins(df: pd.DataFrame) -> pd.DataFrame:
    """Exécuter toutes les jointures dans l'ordre."""
    df = join_iris(df)
    df = join_dpe(df)
    df = join_bpe(df)
    return df


if __name__ == "__main__":
    df = load_dvf_clean()
    df = run_all_joins(df)
    output = DATA_PROCESSED_DIR / "dvf_angers_joined.parquet"
    df.to_parquet(output, index=False)
    logger.info(f"✓ Dataset enrichi sauvegardé : {output.name}")
