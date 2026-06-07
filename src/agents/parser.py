"""
Agent LLM pour extraire les caractéristiques structurées d'une annonce.
Utilise le client OpenAI compatible avec OpenRouter pour parser le texte brut.
"""

import json
import re
from datetime import datetime
from difflib import get_close_matches
from typing import Optional

import pandas as pd
import requests
from openai import OpenAI
from pydantic import BaseModel, Field

from src.agents.scraper import fetch_annonce_text
from src.models.predict import AppartementInput
from src.utils.config import settings
from src.utils.logging import logger

BAN_GEOCODE_URL = "https://api-adresse.data.gouv.fr/search/"
BAN_SCORE_MIN = 0.65  # seuil en dessous duquel le géocodage est rejeté


class AnnonceFeatures(BaseModel):
    """Caractéristiques extraites d'une annonce par l'agent LLM."""

    surface_m2: Optional[float] = Field(None, description="Surface du bien en m²")
    nombre_pieces: Optional[int] = Field(None, description="Nombre de pièces")
    nb_lots_copro: Optional[int] = Field(None, description="Nombre de lots dans la copropriété")
    prix_annonce: Optional[float] = Field(None, description="Prix demandé dans l'annonce en euros")
    dpe_classe: Optional[str] = Field(None, description="Classe DPE (A à G)")
    annee_construction: Optional[int] = Field(None, description="Année de construction")
    quartier: Optional[str] = Field(None, description="Quartier ou adresse")
    adresse: Optional[str] = Field(None, description="Adresse complète si disponible")
    charges_copro: Optional[float] = Field(None, description="Charges de copropriété mensuelles en euros")


PARSE_PROMPT = """
Tu es un expert immobilier. Analyse le texte suivant d'une annonce immobilière
et extrait les informations structurées demandées.

Texte de l'annonce :
{annonce_text}

Réponds UNIQUEMENT avec un JSON valide contenant les champs suivants
(utilise null si l'information est absente) :
{{
    "surface_m2": <float|null>,
    "nombre_pieces": <int|null>,
    "nb_lots_copro": <int|null>,
    "prix_annonce": <float|null>,
    "dpe_classe": <"A"|"B"|"C"|"D"|"E"|"F"|"G"|null>,
    "annee_construction": <int|null>,
    "quartier": <str|null>,
    "adresse": <str|null>,
    "charges_copro": <float|null>
}}
"""


def _extract_json(raw: str) -> str:
    """Extraire le JSON brut depuis une réponse LLM (retire les balises ```json ... ```)."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def geocode_adresse(adresse: str) -> tuple[float, float] | None:
    """
    Géocoder une adresse de rue via l'API BAN.
    Ne pas appeler avec un simple nom de quartier (score trop faible).

    Args:
        adresse: Adresse de rue complète (ex: "12 rue de la Paix, Angers")

    Returns:
        (latitude, longitude) ou None si score < BAN_SCORE_MIN
    """
    try:
        r = requests.get(
            BAN_GEOCODE_URL,
            params={"q": adresse, "limit": 1, "citycode": "49007"},
            timeout=10,
        )
        r.raise_for_status()
        features = r.json().get("features", [])
        if not features:
            logger.warning(f"⚠ Géocodage sans résultat pour : {adresse}")
            return None
        coords = features[0]["geometry"]["coordinates"]
        lon, lat = coords[0], coords[1]
        score = features[0]["properties"].get("score", 0)
        if score < BAN_SCORE_MIN:
            logger.warning(f"⚠ Géocodage rejeté (score={score:.2f} < {BAN_SCORE_MIN}) pour : {adresse}")
            return None
        logger.info(f"✓ Géocodage : {adresse} → ({lat:.5f}, {lon:.5f}) score={score:.2f}")
        return lat, lon
    except requests.RequestException as e:
        logger.warning(f"⚠ Erreur géocodage ({e})")
        return None


class AnnonceParser:
    """Agent LLM pour parser les annonces immobilières."""

    def __init__(self) -> None:
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY manquante dans .env")
        self._client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.llm_base_url,
        )

    def parse_url(self, url: str) -> AnnonceFeatures:
        """
        Scraper une annonce et extraire ses caractéristiques via LLM.

        Args:
            url: URL de l'annonce

        Returns:
            AnnonceFeatures avec les champs extraits
        """
        text = fetch_annonce_text(url)
        if not text:
            logger.warning("Texte vide, retour de features nulles")
            return AnnonceFeatures()

        text_truncated = text[:8000]

        logger.info("Extraction des caractéristiques via LLM...")
        response = self._client.chat.completions.create(
            model=settings.llm_model,
            temperature=0,
            messages=[
                {"role": "user", "content": PARSE_PROMPT.format(annonce_text=text_truncated)},
            ],
        )

        raw = response.choices[0].message.content or ""
        try:
            data = json.loads(_extract_json(raw))
            features = AnnonceFeatures(**data)
            logger.info(f"✓ Caractéristiques extraites : {features}")
            return features
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"✗ Erreur parsing LLM : {e} | réponse brute : {raw[:200]!r}")
            return AnnonceFeatures()

    def to_appart_input(self, features: AnnonceFeatures) -> AppartementInput:
        """
        Convertir les AnnonceFeatures en AppartementInput pour la prédiction.

        Stratégie de géocodage :
        - Si adresse de rue disponible → géocodage BAN (score >= 0.65 requis)
          puis enrichissement spatial (transport, PPRI, IRIS par GPS)
        - Si seulement un quartier → fuzzy match IRIS direct (pas de GPS)

        Args:
            features: Caractéristiques extraites de l'annonce

        Returns:
            AppartementInput utilisable par ImmoPricePredictor
        """
        from src.data.features import ANNEE_REF, DPE_ORDINAL

        dpe_ordinal = DPE_ORDINAL.get(features.dpe_classe) if features.dpe_classe else None
        dpe_bonus = int(features.dpe_classe in ["A", "B"]) if features.dpe_classe else None
        dpe_malus = int(features.dpe_classe in ["F", "G"]) if features.dpe_classe else None
        age_bien = (ANNEE_REF - features.annee_construction) if features.annee_construction else None

        now = datetime.now()

        appart = AppartementInput(
            surface_m2=features.surface_m2 or 50.0,
            nombre_pieces=features.nombre_pieces or 3,
            nb_lots_copro=features.nb_lots_copro or 20,
            mois_vente=now.month,
            annee_vente=now.year,
            dpe_ordinal=dpe_ordinal,
            dpe_bonus=dpe_bonus,
            dpe_malus=dpe_malus,
            age_bien=age_bien,
        )

        if features.adresse:
            # Adresse de rue : géocodage BAN + enrichissement spatial complet
            coords = geocode_adresse(f"{features.adresse}, Angers")
            if coords:
                lat, lon = coords
                appart = _enrich_with_spatial_features(appart, lat, lon, features.quartier)
            elif features.quartier:
                # Géocodage échoué : fallback fuzzy IRIS sur le quartier
                appart = _enrich_iris_from_quartier(appart, features.quartier)
        elif features.quartier:
            # Pas d'adresse de rue : fuzzy match IRIS directement, sans géocodage
            appart = _enrich_iris_from_quartier(appart, features.quartier)

        return appart


def _resolve_code_iris(
    lat: float,
    lon: float,
    quartier: str | None = None,
) -> str | None:
    """
    Résoudre le code_iris d'un bien par sjoin GPS sur les polygones IRIS.
    Fuzzy match en fallback si le sjoin ne donne rien et qu'un quartier est fourni.
    """
    import geopandas as gpd
    from shapely.geometry import Point
    from src.utils.config import DATA_RAW_DIR

    iris_path = DATA_RAW_DIR / "iris" / "contours_iris_49.geojson"
    if not iris_path.exists():
        return None

    iris_geo = gpd.read_file(iris_path)

    # Étape 1 : sjoin GPS
    point = gpd.GeoDataFrame([{"geometry": Point(lon, lat)}], crs=4326)
    joined = gpd.sjoin(point, iris_geo[["code_iris", "nom_iris", "geometry"]], how="left", predicate="within")
    code_iris = joined["code_iris"].iloc[0] if not joined.empty else None
    if pd.notna(code_iris):
        logger.info(f"✓ IRIS résolu par GPS : {code_iris} ({joined['nom_iris'].iloc[0]})")
        return str(code_iris)

    # Étape 2 : fuzzy match
    if quartier:
        return _fuzzy_match_iris(quartier, iris_geo)

    return None


def _fuzzy_match_iris(quartier: str, iris_geo=None) -> str | None:
    """
    Résoudre un code_iris par fuzzy match sur le nom de quartier.

    Args:
        quartier: Nom de quartier extrait de l'annonce
        iris_geo: GeoDataFrame IRIS déjà chargé (optionnel, recharge si None)

    Returns:
        code_iris ou None
    """
    import geopandas as gpd
    from src.utils.config import DATA_RAW_DIR

    if iris_geo is None:
        iris_path = DATA_RAW_DIR / "iris" / "contours_iris_49.geojson"
        if not iris_path.exists():
            return None
        iris_geo = gpd.read_file(iris_path)

    # Nettoyer le quartier : retirer la ville et le code postal en fin
    quartier_clean = re.sub(r",?\s*(Angers|\d{5}).*$", "", quartier, flags=re.IGNORECASE).strip()

    noms = iris_geo["nom_iris"].dropna().tolist()
    matches = get_close_matches(quartier_clean, noms, n=1, cutoff=0.4)
    if matches:
        row = iris_geo[iris_geo["nom_iris"] == matches[0]].iloc[0]
        logger.info(f"✓ IRIS résolu par fuzzy match : '{quartier_clean}' → '{matches[0]}' ({row['code_iris']})")
        return str(row["code_iris"])

    logger.warning(f"⚠ Fuzzy match IRIS sans résultat pour : '{quartier_clean}'")
    return None


def _enrich_iris_from_quartier(
    appart: AppartementInput,
    quartier: str,
) -> AppartementInput:
    """
    Enrichir uniquement les stats IRIS depuis un nom de quartier (sans GPS).
    Utilisé quand l'adresse de rue est absente ou le géocodage a échoué.
    """
    from src.utils.config import DATA_PROCESSED_DIR

    iris_stats_path = DATA_PROCESSED_DIR / "train_test_split" / "iris_stats.parquet"
    if not iris_stats_path.exists():
        logger.warning("⚠ iris_stats.parquet introuvable — relancer make data-features && make train")
        return appart

    code_iris = _fuzzy_match_iris(quartier)
    if not code_iris:
        return appart

    iris_stats = pd.read_parquet(iris_stats_path)
    row = iris_stats[iris_stats["code_iris"] == code_iris]
    if not row.empty:
        object.__setattr__(appart, "prix_m2_median_quartier", float(row["prix_m2_median_quartier"].iloc[0]))
        object.__setattr__(appart, "nb_ventes_quartier", int(row["nb_ventes_quartier"].iloc[0]))
        logger.info(
            f"✓ Stats IRIS (fuzzy) : médiane={row['prix_m2_median_quartier'].iloc[0]:.0f} €/m², "
            f"n={row['nb_ventes_quartier'].iloc[0]} ventes ({code_iris})"
        )
    return appart


def _enrich_with_spatial_features(
    appart: AppartementInput,
    lat: float,
    lon: float,
    quartier: str | None = None,
) -> AppartementInput:
    """
    Enrichir un AppartementInput avec les features spatiales
    calculées depuis les fichiers précalculés (transport, PPRI, IRIS par GPS).
    """
    import geopandas as gpd
    from shapely.geometry import Point
    from src.utils.config import DATA_RAW_DIR, DATA_PROCESSED_DIR

    point_wgs84 = Point(lon, lat)
    EPSG_WGS84 = 4326
    EPSG_LAMBERT = 2154

    # --- Distance tram et gare ---
    transport_path = DATA_RAW_DIR / "transport" / "arrets_transport.parquet"
    if transport_path.exists():
        transport = pd.read_parquet(transport_path)
        gdf_transport = gpd.GeoDataFrame(
            transport,
            geometry=gpd.points_from_xy(transport["lon"], transport["lat"]),
            crs=EPSG_WGS84,
        ).to_crs(EPSG_LAMBERT)
        pt_lambert = gpd.GeoSeries([point_wgs84], crs=EPSG_WGS84).to_crs(EPSG_LAMBERT).iloc[0]
        for transport_type, field in [("tram", "distance_tram_proche_m"), ("gare", "distance_gare_proche_m")]:
            subset = gdf_transport[gdf_transport["type"] == transport_type]
            if not subset.empty:
                object.__setattr__(appart, field, float(subset.geometry.distance(pt_lambert).min()))

    # --- Zone inondable ---
    ppri_path = DATA_RAW_DIR / "ppri" / "zones_inondables_49.geojson"
    if ppri_path.exists():
        ppri = gpd.read_file(ppri_path).to_crs(EPSG_LAMBERT)
        pt_lambert = gpd.GeoSeries([point_wgs84], crs=EPSG_WGS84).to_crs(EPSG_LAMBERT).iloc[0]
        object.__setattr__(appart, "zone_inondable", int(ppri.geometry.contains(pt_lambert).any()))

    # --- Stats IRIS par GPS ---
    iris_stats_path = DATA_PROCESSED_DIR / "train_test_split" / "iris_stats.parquet"
    if iris_stats_path.exists():
        code_iris = _resolve_code_iris(lat, lon, quartier)
        if code_iris:
            iris_stats = pd.read_parquet(iris_stats_path)
            row = iris_stats[iris_stats["code_iris"] == code_iris]
            if not row.empty:
                object.__setattr__(appart, "prix_m2_median_quartier", float(row["prix_m2_median_quartier"].iloc[0]))
                object.__setattr__(appart, "nb_ventes_quartier", int(row["nb_ventes_quartier"].iloc[0]))
                logger.info(
                    f"✓ Stats IRIS (GPS) : médiane={row['prix_m2_median_quartier'].iloc[0]:.0f} €/m², "
                    f"n={row['nb_ventes_quartier'].iloc[0]} ventes ({code_iris})"
                )
    else:
        logger.warning("⚠ iris_stats.parquet introuvable — relancer make data-features && make train")

    return appart
