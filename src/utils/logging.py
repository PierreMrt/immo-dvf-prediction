"""
Configuration centralisée du logging.
"""

import logging

from src.utils.config import LOGS_DIR, settings


def setup_logging() -> logging.Logger:
    """Configurer le logging global (console + fichier)."""

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(settings.log_file)
    file_handler.setLevel(getattr(logging, settings.log_level))
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("immo_dvf")
    logger.setLevel(getattr(logging, settings.log_level))

    # Éviter les handlers dupliqués en cas de rechargement
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# Logger global importé dans tous les modules
logger = setup_logging()
