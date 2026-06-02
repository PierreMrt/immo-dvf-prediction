"""
Prédiction du prix au m² pour un appartement donné.
Utilisé par l'application Streamlit et les agents.
"""

import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from src.utils.config import MODELS_DIR, load_metadata
from src.utils.logging import logger


@dataclass
class AppartementInput:
    """Caractéristiques d'un appartement pour la prédiction."""

    surface_m2: float
    nombre_pieces: int
    nb_lots_copro: int = 20
    mois_vente: int = 6
    annee_vente: int = 2026
    # DPE (optionnel)
    dpe_ordinal: Optional[int] = None
    dpe_bonus: Optional[int] = None
    dpe_malus: Optional[int] = None
    age_bien: Optional[int] = None
    # Quartier (optionnel)
    prix_m2_median_quartier: Optional[float] = None
    nb_ventes_quartier: Optional[int] = None
    prix_m2_vs_quartier: Optional[float] = None
    # Équipements (optionnel)
    nb_commerces_500m: Optional[int] = None
    nb_restaurants_500m: Optional[int] = None
    nb_ecoles_500m: Optional[int] = None
    nb_parcs_500m: Optional[int] = None
    score_commodites: Optional[int] = None


@dataclass
class PredictionResult:
    """Résultat d'une prédiction de prix."""

    prix_m2_predit: float
    prix_total_predit: float
    surface_m2: float
    features_utilisees: list[str]


class ImmoPricePredictor:
    """Prédicteur de prix immobiliers pour appartements Angers."""

    def __init__(self) -> None:
        self._model = None
        self._feature_columns: list[str] = []
        self._load_model()

    def _load_model(self) -> None:
        """Charger le modèle et les colonnes depuis le disque."""
        metadata = load_metadata()
        if not metadata:
            raise FileNotFoundError("Aucun modèle trouvé. Lancer make train.")

        model_path = MODELS_DIR / metadata["model_file"]
        self._model = joblib.load(model_path)
        self._feature_columns = metadata["features"]
        logger.info(f"Modèle chargé : {model_path.name}")

    def predict(self, appart: AppartementInput) -> PredictionResult:
        """
        Prédire le prix au m² d'un appartement.

        Args:
            appart: Caractéristiques de l'appartement

        Returns:
            PredictionResult avec prix prédit et métadonnées
        """
        # Construire le vecteur de features
        data = {
            "surface_m2": appart.surface_m2,
            "nombre_pieces": appart.nombre_pieces,
            "surface_par_piece": appart.surface_m2 / appart.nombre_pieces,
            "nb_lots_copro": appart.nb_lots_copro,
            "mois_vente": appart.mois_vente,
            "annee_vente": appart.annee_vente,
            "copro_petite": int(appart.nb_lots_copro < 10),
            "copro_grande": int(appart.nb_lots_copro > 50),
        }

        # Ajouter les features optionnelles si disponibles
        optional_fields = [
            "dpe_ordinal", "dpe_bonus", "dpe_malus", "age_bien",
            "prix_m2_median_quartier", "nb_ventes_quartier", "prix_m2_vs_quartier",
            "nb_commerces_500m", "nb_restaurants_500m", "nb_ecoles_500m",
            "nb_parcs_500m", "score_commodites",
        ]
        for field in optional_fields:
            val = getattr(appart, field)
            if val is not None:
                data[field] = val

        # Aligner sur les colonnes du modèle
        X = pd.DataFrame([data])
        X = X.reindex(columns=self._feature_columns, fill_value=np.nan)

        prix_m2 = float(self._model.predict(X)[0])
        prix_total = prix_m2 * appart.surface_m2

        return PredictionResult(
            prix_m2_predit=round(prix_m2, 0),
            prix_total_predit=round(prix_total, 0),
            surface_m2=appart.surface_m2,
            features_utilisees=list(X.columns),
        )


if __name__ == "__main__":
    predictor = ImmoPricePredictor()
    exemple = AppartementInput(
        surface_m2=55,
        nombre_pieces=3,
        dpe_ordinal=4,
        age_bien=30,
        prix_m2_median_quartier=2500,
    )
    result = predictor.predict(exemple)
    print(f"Prix prédit : €{result.prix_total_predit:,.0f} (€{result.prix_m2_predit:,.0f}/m²)")
