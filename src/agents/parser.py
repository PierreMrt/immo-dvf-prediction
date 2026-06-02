"""
Agent LLM pour extraire les caractéristiques structurées d'une annonce.
Utilise LangChain + OpenAI pour parser le texte brut de l'annonce.
"""

import json
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

from src.agents.scraper import fetch_annonce_text
from src.models.predict import AppartementInput
from src.utils.config import settings
from src.utils.logging import logger


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


PARSE_PROMPT = PromptTemplate(
    input_variables=["annonce_text"],
    template="""
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
""",
)


class AnnonceParser:
    """Agent LLM pour parser les annonces immobilières."""

    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY manquante dans .env")
        self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.openai_api_key)
        self._chain = PARSE_PROMPT | self._llm

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

        # Tronquer le texte si trop long (4000 tokens max)
        text_truncated = text[:8000]

        logger.info("Extraction des caractéristiques via LLM...")
        response = self._chain.invoke({"annonce_text": text_truncated})

        try:
            data = json.loads(response.content)
            features = AnnonceFeatures(**data)
            logger.info(f"✓ Caractéristiques extraites : {features}")
            return features
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"✗ Erreur parsing LLM : {e}")
            return AnnonceFeatures()

    def to_appart_input(self, features: AnnonceFeatures) -> AppartementInput:
        """
        Convertir les AnnonceFeatures en AppartementInput pour la prédiction.

        Args:
            features: Caractéristiques extraites de l'annonce

        Returns:
            AppartementInput utilisable par ImmoPricePredictor
        """
        from src.data.features import DPE_ORDINAL, ANNEE_REF

        dpe_ordinal = DPE_ORDINAL.get(features.dpe_classe) if features.dpe_classe else None
        dpe_bonus = int(features.dpe_classe in ["A", "B"]) if features.dpe_classe else None
        dpe_malus = int(features.dpe_classe in ["F", "G"]) if features.dpe_classe else None
        age_bien = (ANNEE_REF - features.annee_construction) if features.annee_construction else None

        return AppartementInput(
            surface_m2=features.surface_m2 or 50.0,
            nombre_pieces=features.nombre_pieces or 3,
            nb_lots_copro=features.nb_lots_copro or 20,
            dpe_ordinal=dpe_ordinal,
            dpe_bonus=dpe_bonus,
            dpe_malus=dpe_malus,
            age_bien=age_bien,
        )
