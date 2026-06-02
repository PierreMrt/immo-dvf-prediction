"""
Scraping des annonces immobilières.
Extraire le texte brut d'une URL d'annonce pour l'envoyer à l'agent LLM.
"""

import requests
from bs4 import BeautifulSoup

from src.utils.logging import logger

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_annonce_text(url: str) -> str:
    """
    Récupérer le contenu textuel brut d'une annonce immobilière.

    Le texte est ensuite passé à l'agent LLM pour extraction structurée.

    Args:
        url: URL de l'annonce (SeLoger, Bien'Ici, LeBonCoin...)

    Returns:
        Texte brut de la page (nettoyé)
    """
    logger.info(f"Scraping annonce : {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "lxml")

        # Supprimer les balises inutiles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        # Nettoyage basique : espaces multiples
        import re
        text = re.sub(r"\s+", " ", text).strip()

        logger.info(f"✓ Texte extrait ({len(text)} caractères)")
        return text

    except requests.RequestException as e:
        logger.error(f"✗ Erreur scraping {url} : {e}")
        return ""
