"""
Jointures entre DVF et sources externes : DPE, OSM (équipements), IRIS.
"""

import re

import geopandas as gpd
import pandas as pd

from src.data.load import load_dpe, load_dvf_clean, load_equipements_osm
from src.utils.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.utils.logging import logger

RAYON_EQUIPEMENTS_M = 500
EPSG_WGS84 = 4326
EPSG_LAMBERT = 2154

# Abréviations courantes dans les adresses DVF → forme longue pour rapprochement
_ABREV = {
    r"\bBD\b": "BOULEVARD",
    r"\bAV\b": "AVENUE",
    r"\bPL\b": "PLACE",
    r"\bIMP\b": "IMPASSE",
    r"\bSQ\b": "SQUARE",
    r"\bRES\b": "RESIDENCE",
    r"\bST\b": "SAINT",
    r"\bSTE\b": "SAINTE",
    r"\bALL\b": "ALLEE",
    r"\bCHE\b": "CHEMIN",
    r"\bLOT\b": "LOTISSEMENT",
}


def _normalize_address(s: pd.Series) -> pd.Series:
    """
    Normaliser une série d'adresses :
    - Majuscules, strip
    - Suppression accents (via unicode normalize)
    - Remplacement abréviations DVF
    - Suppression caractères non alphanumériques
    """
    import unicodedata
    s = (
        s.fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )
    # Suppression accents
    s = s.apply(
        lambda x: unicodedata.normalize("NFD", x)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    # Expansion abréviations
    for pattern, replacement in _ABREV.items():
        s = s.str.replace(pattern, replacement, regex=True)
    # Garder uniquement alphanumériques + espaces
    s = s.str.replace(r"[^A-Z0-9 ]", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def _parse_adresse_ban(adresse_ban: pd.Series) -> pd.DataFrame:
    """
    Extraire numéro et nom de voie depuis adresse_ban DPE.
    Format observé : "35 Square des Anciennes Provinces 49000 Angers"
    On supprime la partie code postal + ville en fin de chaîne.
    """
    # Supprimer " CPPPP Ville" en fin de chaîne
    cleaned = adresse_ban.str.upper().str.strip()
    cleaned = cleaned.str.replace(r"\s+\d{5}\s+\S.*$", "", regex=True).str.strip()
    # Séparer numéro (optionnel) du reste
    numero = cleaned.str.extract(r"^(\d+)", expand=False).fillna("")
    nom_voie = cleaned.str.replace(r"^\d+\s*", "", regex=True).str.strip()
    return pd.DataFrame({"dpe_numero": numero, "dpe_nom_voie": nom_voie})


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

    Stratégie :
    1. Parser adresse_ban DPE pour extraire numéro + nom voie
    2. Normaliser les deux côtés (majuscules, sans accents, abréviations étendues)
    3. Clé de jointure : numéro + nom voie normalisés
    """
    logger.info("Jointure DPE...")
    dpe = load_dpe()

    # Parser + normaliser adresses DPE
    dpe_parsed = _parse_adresse_ban(dpe["adresse_ban"])
    dpe["_num"] = dpe_parsed["dpe_numero"]
    dpe["_voie"] = _normalize_address(dpe_parsed["dpe_nom_voie"])
    dpe["_cle"] = dpe["_num"] + " " + dpe["_voie"]
    dpe["_cle"] = dpe["_cle"].str.strip()

    df = df.copy()
    num_col = next((c for c in ["adresse_numero", "numero_voie"] if c in df.columns), None)
    voie_col = next((c for c in ["adresse_nom_voie", "nom_voie"] if c in df.columns), None)

    if not (num_col and voie_col):
        logger.warning("⚠ Colonnes d'adresse DVF introuvables, jointure DPE ignorée")
        return df

    # Normaliser adresses DVF
    num_str = df[num_col].fillna(0).astype(float).astype(int).astype(str)
    num_str = num_str.replace("0", "")
    df["_cle"] = (num_str + " " + _normalize_address(df[voie_col])).str.strip()

    dpe_cols = [
        "_cle",
        "etiquette_dpe",
        "etiquette_ges",
        "annee_construction",
        "conso_5_usages_par_m2_ep",
        "emission_ges_5_usages_par_m2",
    ]
    dpe_cols_available = [c for c in dpe_cols if c in dpe.columns]

    merged = df.merge(
        dpe[dpe_cols_available].drop_duplicates("_cle"),
        on="_cle",
        how="left",
    )

    n_matched = merged["etiquette_dpe"].notna().sum() if "etiquette_dpe" in merged.columns else 0
    logger.info(f"✓ DPE jointé : {n_matched:,} / {len(df):,} ({n_matched / len(df):.1%})")
    return merged.drop(columns=["_cle"])


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
