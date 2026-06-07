"""
Agent LLM pour extraire les caractéristiques structurées d'une annonce.
Utilise le client OpenAI compatible avec OpenRouter pour parser le texte brut.
"""

import json
import re
from datetime import datetime
from typing import Optional

import requests
from openai import OpenAI
from pydantic import BaseModel, Field

from src.agents.scraper import fetch_annonce_text
from src.models.predict import AppartementInput
from src.utils.config import settings
from src.utils.logging import logger

BAN_GEOCODE_URL = "https://api-adresse.data.gouv.fr/search/"


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
    Géocoder une adresse via l'API Adresse data.gouv.fr (BAN).

    Args:
        adresse: Adresse textuelle (ex: "12 rue de la Paix, Angers")

    Returns:
        (latitude, longitude) ou None si échec
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
        if score < 0.5:
            logger.warning(f"⚠ Géocodage de faible confiance (score={score:.2f}) pour : {adresse}")
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
        Inclut le géocodage de l'adresse pour alimenter les features spatiales.

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

        # Géocodage → features spatiales
        adresse = features.adresse or features.quartier
        if adresse:
            coords = geocode_adresse(f"{adresse}, Angers")
            if coords:
                lat, lon = coords
                appart = _enrich_with_spatial_features(appart, lat, lon)

        return appart


def _enrich_with_spatial_features(
    appart: AppartementInput,
    lat: float,
    lon: float,
) -> AppartementInput:
    """
    Enrichir un AppartementInput avec les features spatiales
    calculées depuis les fichiers précalculés (transport, PPRI, stats IRIS).

    Args:
        appart: Input à enrichir
        lat: Latitude WGS-84
        lon: Longitude WGS-84

    Returns:
        AppartementInput enrichi (nouvelles valeurs si fichiers disponibles)
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
        import pandas as pd
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
                dist = float(subset.geometry.distance(pt_lambert).min())
                object.__setattr__(appart, field, dist)

    # --- Zone inondable ---
    ppri_path = DATA_RAW_DIR / "ppri" / "zones_inondables_49.geojson"
    if ppri_path.exists():
        ppri = gpd.read_file(ppri_path).to_crs(EPSG_LAMBERT)
        pt_lambert = gpd.GeoSeries([point_wgs84], crs=EPSG_WGS84).to_crs(EPSG_LAMBERT).iloc[0]
        in_flood_zone = int(ppri.geometry.contains(pt_lambert).any())
        object.__setattr__(appart, "zone_inondable", in_flood_zone)

    # --- Stats quartier depuis le dataset features précalculé ---
    features_path = DATA_PROCESSED_DIR / "dvf_angers_features.parquet"
    iris_path = DATA_RAW_DIR / "iris" / "contours_iris_49.geojson"
    if features_path.exists() and iris_path.exists():
        import pandas as pd
        iris_geo = gpd.read_file(iris_path)
        gdf_pt = gpd.GeoDataFrame(
            [{"geometry": point_wgs84}], crs=EPSG_WGS84
        )
        joined = gpd.sjoin(gdf_pt, iris_geo[["code_iris", "geometry"]], how="left", predicate="within")
        code_iris = joined["code_iris"].iloc[0] if not joined.empty else None

        if code_iris:
            df_feat = pd.read_parquet(features_path, columns=["code_iris", "prix_m2"])
            iris_stats = df_feat[df_feat["code_iris"] == code_iris]["prix_m2"]
            if not iris_stats.empty:
                object.__setattr__(appart, "prix_m2_median_quartier", float(iris_stats.median()))
                object.__setattr__(appart, "nb_ventes_quartier", int(len(iris_stats)))

    return appart
