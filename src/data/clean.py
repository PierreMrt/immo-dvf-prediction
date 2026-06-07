"""
Nettoyage des données DVF brutes.
Filtre Angers (49007), appartements, ventes valides, outliers.
"""

import unicodedata

import pandas as pd

from src.data.load import load_dvf_raw
from src.utils.config import DATA_PROCESSED_DIR, settings
from src.utils.logging import logger

# Seuils de filtrage outliers (prix total et surface)
PRIX_MIN = 10_000
PRIX_MAX = 2_000_000
SURFACE_MIN = 9
SURFACE_MAX = 500

# Seuils de filtrage outliers sur le prix au m²
# Angers : marché entre ~1 000 et ~8 000 €/m² (hors biens atypiques)
PRIX_M2_MIN = 1_000
PRIX_M2_MAX = 8_000

# Abréviations DVF → forme longue (ordre important : plus long en premier)
_ABREV_DVF = [
    ("CITE",       "CITE"),       # inchangé mais préserve du remplacement parasite
    ("CHE",        "CHEMIN"),
    ("IMP",        "IMPASSE"),
    ("LOT",        "LOTISSEMENT"),
    ("RES",        "RESIDENCE"),
    ("STE",        "SAINTE"),
    ("ALL",        "ALLEE"),
    ("AVE",        "AVENUE"),
    ("AV",         "AVENUE"),
    ("BD",         "BOULEVARD"),
    ("BLD",        "BOULEVARD"),
    ("BLVD",       "BOULEVARD"),
    ("PL",         "PLACE"),
    ("SQ",         "SQUARE"),
    ("ST",         "SAINT"),
    ("HAM",        "HAMEAU"),
    ("VLA",        "VILLA"),
    ("DOM",        "DOMAINE"),
    ("PAR",        "PARC"),
    ("SENT",       "SENTIER"),
    ("LD",         "LIEU DIT"),
    ("ABBE",       "ABBE"),       # déjà sans accent, on garde
]


def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii")


def normalize_adresse(numero: pd.Series, nom_voie: pd.Series) -> pd.Series:
    """
    Construire une clé d'adresse normalisée depuis les colonnes DVF.

    Pipeline :
    1. Numéro : float → int → str ("9.0" → "9"), vide si NaN/0
    2. Voie  : majuscules, sans accents, expansion abréviations
    3. Concaténation + nettoyage espaces
    """
    # Numéro
    num = (
        numero.fillna(0).astype(float).astype(int)
        .astype(str).replace("0", "")
    )

    # Voie : majuscules + sans accents
    voie = (
        nom_voie.fillna("").astype(str)
        .str.upper()
        .apply(_strip_accents)
    )

    # Expansion abréviations mot par mot
    abrev_map = {k: v for k, v in _ABREV_DVF}

    def _expand(s: str) -> str:
        tokens = s.split()
        return " ".join(abrev_map.get(t, t) for t in tokens)

    voie = voie.apply(_expand)

    # Suppression caractères non alphanumériques
    voie = voie.str.replace(r"[^A-Z0-9 ]", "", regex=True)
    voie = voie.str.replace(r"\s+", " ", regex=True).str.strip()

    return (num + " " + voie).str.strip()


def normalize_adresse_ban(adresse_ban: pd.Series) -> pd.Series:
    """
    Normaliser les adresses DPE (format BAN : "35 Rue Truc 49000 Angers").

    Pipeline :
    1. Supprimer " CPPPP Ville" en fin de chaîne
    2. Majuscules + sans accents
    3. Suppression caractères non alphanumériques
    """
    s = adresse_ban.fillna("").astype(str).str.upper()
    s = s.apply(_strip_accents)
    # Supprimer CP + ville en fin : " 49000 ANGERS" ou " 49070 BEAUCOUZE"
    s = s.str.replace(r"\s+\d{5}(?:\s+\S.*)?$", "", regex=True).str.strip()
    # Nettoyage
    s = s.str.replace(r"[^A-Z0-9 ]", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def clean_dvf_angers() -> pd.DataFrame:
    """
    Nettoyer les données DVF pour Angers (appartements uniquement).

    Returns:
        DataFrame nettoyé avec features DVF de base
    """
    df = load_dvf_raw()

    logger.info("Filtrage Angers + Appartements + Ventes valides...")
    df = df[
        (df["code_commune"] == settings.city_code_insee)
        & (df["type_local"] == "Appartement")
        & (df["nature_mutation"] == "Vente")
        & (df["latitude"].notna())
        & (df["longitude"].notna())
        & (df["surface_reelle_bati"].between(SURFACE_MIN, SURFACE_MAX))
        & (df["valeur_fonciere"].between(PRIX_MIN, PRIX_MAX))
    ].reset_index(drop=True)

    logger.info(f"Après filtrage initial : {len(df):,} ventes")

    # Renommer les colonnes pour la clarté
    df = df.rename(
        columns={
            "surface_reelle_bati": "surface_m2",
            "valeur_fonciere": "prix_vente",
            "nombre_pieces_principales": "nombre_pieces",
            "nombre_lots": "nb_lots_copro",
            "date_mutation": "date_vente",
        }
    )

    # Features temporelles de base
    df["prix_m2"] = df["prix_vente"] / df["surface_m2"]
    df["mois_vente"] = df["date_vente"].dt.month
    df["annee_vente"] = df["date_vente"].dt.year

    # Filtre outliers prix au m² (après calcul)
    avant = len(df)
    df = df[df["prix_m2"].between(PRIX_M2_MIN, PRIX_M2_MAX)].reset_index(drop=True)
    logger.info(f"Filtre prix/m² [{PRIX_M2_MIN}–{PRIX_M2_MAX} €/m²] : {avant - len(df):,} outliers supprimés → {len(df):,} ventes conservées")

    # Adresse normalisée pour jointure DPE
    df["adresse_norm"] = normalize_adresse(
        df["adresse_numero"], df["adresse_nom_voie"]
    )
    logger.info("Adresse normalisée ajoutée (adresse_norm)")

    output_path = DATA_PROCESSED_DIR / "dvf_angers_appart_clean.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"✓ DVF nettoyé sauvegardé : {output_path}")

    return df


if __name__ == "__main__":
    clean_dvf_angers()
