"""
Configuration centralisée du projet.
Utilise pydantic-settings pour valider les variables d'environnement.
"""

import json
from pathlib import Path
from typing import Any, List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Configuration principale du projet."""

    # App info
    app_name: str = Field("immo-dvf-prediction")
    app_version: str = Field("0.1.0")

    # Chemins de données
    data_raw_dir: Path = Field(Path("data/raw"))
    data_processed_dir: Path = Field(Path("data/processed"))
    models_dir: Path = Field(Path("models"))

    # dvf_years stocké en str pour éviter le pré-parsing JSON de pydantic-settings.
    # La propriété dvf_years_parsed expose la liste d'entiers.
    dvf_years: str = Field(default="2024,2025,2026")

    city_code_insee: str = Field("49007")
    department_code: str = Field("49")

    # Configuration ML
    random_seed: int = Field(42)
    train_test_split: float = Field(0.8)
    default_model: str = Field("xgboost")

    # Logging
    log_level: str = Field("INFO")
    log_file: Path = Field(Path("logs/app.log"))

    # API Keys
    openai_api_key: Optional[str] = Field(None)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def dvf_years_list(self) -> List[int]:
        """Retourner dvf_years sous forme de liste d'entiers."""
        v = self.dvf_years.strip()
        if v.startswith("["):
            return [int(x) for x in json.loads(v)]
        return [int(x.strip()) for x in v.split(",") if x.strip()]


# Instance singleton des settings
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
