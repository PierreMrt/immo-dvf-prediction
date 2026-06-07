"""
Scraping des annonces immobilières via Playwright.

Playwright lance un vrai navigateur Chromium headless, contournant
le fingerprinting TLS et les anti-bots (Cloudflare, leboncoin...).

Dépendances : playwright (pip install playwright && playwright install chromium)
"""

import re

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

from src.utils.logging import logger

# Délai max (ms) pour le chargement de la page
_PAGE_TIMEOUT = 30_000
# Sélecteurs à supprimer avant extraction du texte
_TAGS_TO_REMOVE = ["script", "style", "nav", "footer", "header"]


def fetch_annonce_text(url: str) -> str:
    """
    Récupérer le contenu textuel brut d'une annonce immobilière.

    Utilise Playwright (Chromium headless) pour contourner les protections
    anti-bot des sites (Cloudflare, TLS fingerprinting, leboncoin...).
    Le texte est ensuite passé à l'agent LLM pour extraction structurée.

    Args:
        url: URL de l'annonce (SeLoger, Bien'Ici, LeBonCoin...)

    Returns:
        Texte brut de la page (nettoyé), chaîne vide si échec.
    """
    logger.info(f"Scraping annonce : {url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                locale="fr-FR",
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
            )
            page = context.new_page()

            page.goto(url, wait_until="domcontentloaded", timeout=_PAGE_TIMEOUT)

            # Supprimer les balises inutiles avant extraction
            for tag in _TAGS_TO_REMOVE:
                page.evaluate(f"""
                    document.querySelectorAll('{tag}').forEach(el => el.remove())
                """)

            text = page.inner_text("body")
            browser.close()

        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            logger.warning("Texte vide, retour de features nulles")
            return ""

        logger.info(f"✓ Texte extrait ({len(text)} caractères)")
        return text

    except PlaywrightTimeout:
        logger.error(f"✗ Timeout Playwright ({_PAGE_TIMEOUT}ms) : {url}")
        return ""
    except Exception as e:
        logger.error(f"✗ Erreur scraping {url} : {e}")
        return ""
