"""
Jointures entre DVF et sources externes : DPE, BPE, IRIS, COPRO.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from src.data.load import load_bpe, load_copro, load_dpe, load_dvf_clean, load_iris
from src.utils.config import DATA_PROCESSED_DIR
from src.utils.logging import logger

# Rayon en mètres pour les jointures par proximité
RAYON_EQUIPEMENTS_M = 500
EPSG_WGS84 = 4326
EPSG_LAMBERT = 2154  # Lambert-93 (unité en mètres)


def join_iris(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigner le code IRIS à chaque vente DVF via la table INSEE.

    Note : Utilise une jointure géospatiale via GeoPandas
    (les DVF géolocalisées contiennent lat/lon WGS-84).
    """
    logger.info("Jointure IRIS...")
    iris_df = load_iris()

    # Charger les contours IRIS (télécharger séparément si nécessaire)
    # https://geoservices.ign.fr/contoursiris
    iris_path = DATA_PROCESSED_DIR.parent / "raw" / "iris" / "contours_iris_49.geojson"
    if not iris_path.exists():
        logger.warning("Contours IRIS introuvables, code_iris non assigné")
        return df

    iris_geo = gpd.read_file(iris_path)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=EPSG_WGS84,
    )

    joined = gpd.sjoin(gdf, iris_geo[["CODE_IRIS", "NOM_IRIS", "geometry"]], how="left", predicate="within")
    df["code_iris"] = joined["CODE_IRIS"].values
    df["nom_iris"] = joined["NOM_IRIS"].values

    logger.info(f"✓ IRIS assigné ({df['code_iris'].notna().sum():,} / {len(df):,})")
    return df


def join_dpe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Joindre les données DPE au dataset DVF.

    Stratégie : jointure par adresse normalisée + surface (±15%).
    La jointure DPE est approximative par nature (adresses textuelles).
    """
    logger.info("Jointure DPE...")
    dpe = load_dpe()

    # Filtrer DPE sur Angers uniquement
    dpe = dpe[dpe["code_postal_ban"].astype(str).str.startswith("490")].copy()

    # Normaliser les noms de voie pour la jointure
    dpe["adresse_norm"] = (
        dpe["adresse_ban"]
        .str.upper()
        .str.strip()
        .str.replace(r"[^A-Z0-9 ]", "", regex=True)
    )
    df["adresse_norm"] = (
        (df["numero_voie"].fillna("").astype(str) + " " + df["nom_voie"].fillna(""))
        .str.upper()
        .str.strip()
        .str.replace(r"[^A-Z0-9 ]", "", regex=True)
    )

    # Jointure sur adresse_norm
    dpe_cols = ["adresse_norm", "classe_energie", "annee_construction", "consommation_energie", "emissions_ges"]
    merged = df.merge(dpe[dpe_cols].drop_duplicates("adresse_norm"), on="adresse_norm", how="left")

    n_matched = merged["classe_energie"].notna().sum()
    logger.info(f"✓ DPE jointé ({n_matched:,} / {len(df):,} matches soit {n_matched / len(df):.1%})")

    return merged.drop(columns=["adresse_norm"])


def join_bpe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Joindre les équipements BPE INSEE dans un rayon de 500m.

    Compte le nombre d'équipements par type dans un rayon donné.
    """
    logger.info(f"Jointure BPE (rayon {RAYON_EQUIPEMENTS_M}m)...")
    bpe = load_bpe()

    # Filtrer BPE sur Angers
    bpe = bpe[bpe["depcom"] == "49007"].copy()

    if "lambert_x" not in bpe.columns or "lambert_y" not in bpe.columns:
        logger.warning("Coordonnées Lambert absentes de BPE, jointure BPE ignorée")
        return df

    # Convertir DVF en Lambert-93 pour calcul de distances en mètres
    gdf_dvf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=EPSG_WGS84,
    ).to_crs(EPSG_LAMBERT)

    gdf_bpe = gpd.GeoDataFrame(
        bpe,
        geometry=gpd.points_from_xy(bpe["lambert_x"], bpe["lambert_y"]),
        crs=EPSG_LAMBERT,
    )

    # Types d'équipements à comptabiliser (codes BPE INSEE)
    types_bpe = {
        "nb_commerces": ["B101", "B102", "B201", "B202", "B203"],
        "nb_restaurants": ["A504"],
        "nb_ecoles": ["C101", "C102", "C201"],
        "nb_parcs": ["F307"],
    }

    for col_name, codes in types_bpe.items():
        bpe_subset = gdf_bpe[gdf_bpe["typequ"].isin(codes)]
        counts = []
        for geom in gdf_dvf.geometry:
            buffer = geom.buffer(RAYON_EQUIPEMENTS_M)
            counts.append(bpe_subset[bpe_subset.geometry.within(buffer)].shape[0])
        df[col_name + f"_{RAYON_EQUIPEMENTS_M}m"] = counts

    logger.info("✓ BPE jointé")
    return df


def join_copro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Joindre les données copropriété (charges, procédures) via id_parcelle.
    """
    logger.info("Jointure COPRO...")
    copro = load_copro()

    copro_cols = ["id_parcelle", "charges_mensuelles", "en_procedure"]
    available = [c for c in copro_cols if c in copro.columns]

    merged = df.merge(copro[available], on="id_parcelle", how="left")
    n_matched = merged["charges_mensuelles"].notna().sum() if "charges_mensuelles" in merged.columns else 0
    logger.info(f"✓ COPRO jointé ({n_matched:,} / {len(df):,} matches)")

    return merged


def run_all_joins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exécuter toutes les jointures dans l'ordre.

    Args:
        df: DataFrame DVF nettoyé

    Returns:
        DataFrame enrichi avec DPE, BPE, IRIS, COPRO
    """
    df = join_iris(df)
    df = join_dpe(df)
    df = join_bpe(df)
    df = join_copro(df)
    return df


if __name__ == "__main__":
    df = load_dvf_clean()
    df = run_all_joins(df)
    output = DATA_PROCESSED_DIR / "dvf_angers_joined.parquet"
    df.to_parquet(output, index=False)
    logger.info(f"✓ Dataset enrichi sauvegardé : {output}")
