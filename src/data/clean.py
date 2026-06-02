"""
Nettoyage des données DVF brutes.
Filtre Angers (49007), appartements, ventes valides, outliers.
"""

import pandas as pd

from src.data.load import load_dvf_raw
from src.utils.config import DATA_PROCESSED_DIR, settings
from src.utils.logging import logger

# Seuils de filtrage outliers (prix et surface)
PRIX_MIN = 10_000
PRIX_MAX = 2_000_000
SURFACE_MIN = 9
SURFACE_MAX = 500


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

    logger.info(f"Après filtrage : {len(df):,} ventes")

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

    output_path = DATA_PROCESSED_DIR / "dvf_angers_appart_clean.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"✓ DVF nettoyé sauvegardé : {output_path}")

    return df


if __name__ == "__main__":
    clean_dvf_angers()
