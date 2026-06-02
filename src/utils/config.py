"""
Configuration centralisée du projet.
Utilise pydantic-settings pour valider les variables d'environnement.
"""

import json
from pathlib import Path
from typing import Any, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Configuration principale du projet."""

    # App info
    app_name: str = Field("immo-dvf-prediction", description="Nom de l'application")
    app_version: str = Field("0.1.0", description="Version de l'application")

    # Chemins de données
    data_raw_dir: Path = Field(Path("data/raw"), description="Répertoire données brutes")
    data_processed_dir: Path = Field(
        Path("data/processed"), description="Répertoire données processées"
    )
    models_dir: Path = Field(Path("models"), description="Répertoire modèles ML")

    # Configuration données
    # Accepte "2024,2025,2026" (CSV) ou "[2024,2025,2026]" (JSON) depuis le .env
    dvf_years: List[int] = Field(
        default=[2024, 2025, 2026], description="Années DVF à télécharger"
    )
    city_code_insee: str = Field("49007", description="Code INSEE d'Angers")
    department_code: str = Field("49", description="Code département Maine-et-Loire")

    # Configuration ML
    random_seed: int = Field(42, description="Seed pour reproductibilité")
    train_test_split: float = Field(0.8, description="Ratio train/test")
    default_model: str = Field("xgboost", description="Modèle par défaut")

    # Logging
    log_level: str = Field("INFO", description="Niveau de logging")
    log_file: Path = Field(Path("logs/app.log"), description="Fichier de logs")

    # API Keys
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")

    @field_validator("dvf_years", mode="before")
    @classmethod
    def parse_dvf_years(cls, v: Any) -> List[int]:
        """
        Accepte trois formats depuis le .env :
          - liste Python déjà déserialisée : [2024, 2025, 2026]
          - JSON string            : "[2024,2025,2026]"
          - CSV string             : "2024,2025,2026"
        """
        if isinstance(v, list):
            return [int(x) for x in v]
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                return [int(x) for x in json.loads(v)]
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        return v

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Instance singleton des settings
settings = Settings()

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
