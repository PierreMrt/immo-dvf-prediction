"""
Configuration centralisée du projet.
Utilise pydantic-settings pour valider les variables d'environnement.
"""

import json
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration principale du projet."""

    # App info
    app_name: str = Field("immo-dvf-prediction")
    app_version: str = Field("0.1.0")

    # Chemins de données (relatifs à la racine du projet)
    data_raw_dir: Path = Field(Path("data/raw"))
    data_processed_dir: Path = Field(Path("data/processed"))
    models_dir: Path = Field(Path("models"))

    # dvf_years stocké en str pour éviter le pré-parsing JSON de pydantic-settings.
    # Utiliser dvf_years_list pour obtenir la liste d'entiers.
    dvf_years: str = Field(default="2024,2025,2026")

    city_code_insee: str = Field("49007")
    department_code: str = Field("49")

    # Configuration ML
    random_seed: int = Field(42)
    # test_size : fraction du dataset réservée au test (ex: 0.2 = 20% test, 80% train)
    test_size: float = Field(0.2)
    default_model: str = Field("xgboost")

    # Logging
    log_level: str = Field("INFO")
    log_file: Path = Field(Path("logs/app.log"))

    # API Keys
    openrouter_api_key: Optional[str] = Field(None)

    # LLM — modèle gratuit OpenRouter (mistral-7b-instruct supprimé en 2026)
    llm_model: str = Field("meta-llama/llama-3.3-70b-instruct:free")
    llm_base_url: str = Field("https://openrouter.ai/api/v1")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

    @property
    def dvf_years_list(self) -> List[int]:
        """Retourner dvf_years sous forme de liste d'entiers."""
        v = self.dvf_years.strip()
        if v.startswith("["):
            return [int(x) for x in json.loads(v)]
        return [int(x.strip()) for x in v.split(",") if x.strip()]


# Instance singleton
settings = Settings()

# Alias pratique utilisé dans tout le projet
DVF_YEARS: List[int] = settings.dvf_years_list

# Chemins absolus depuis la racine du projet
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = ROOT_DIR / settings.data_raw_dir
DATA_PROCESSED_DIR = ROOT_DIR / settings.data_processed_dir
MODELS_DIR = ROOT_DIR / settings.models_dir
LOGS_DIR = ROOT_DIR / "logs"


def save_metadata(metadata: dict, filename: str = "metadata.json") -> None:
    """Sauvegarder les métadonnées du modèle au format JSON."""
    filepath = MODELS_DIR / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)


def load_metadata(filename: str = "metadata.json") -> dict:
    """Charger les métadonnées du modèle depuis un fichier JSON."""
    filepath = MODELS_DIR / filename
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
