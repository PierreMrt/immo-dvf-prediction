"""
Évaluation du modèle entraîné : métriques, résidus, feature importance.
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR, load_metadata
from src.utils.logging import logger


def evaluate_model() -> dict:
    """
    Charger le modèle et le split test, calculer et afficher les métriques.

    Returns:
        Dictionnaire des métriques
    """
    split_dir = DATA_PROCESSED_DIR / "train_test_split"
    X_test = pd.read_parquet(split_dir / "X_test.parquet")
    y_test = pd.read_parquet(split_dir / "y_test.parquet").squeeze()

    metadata = load_metadata()
    if not metadata:
        raise FileNotFoundError("metadata.json introuvable. Lancer make train.")

    model_path = MODELS_DIR / metadata["model_file"]
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    metrics = {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4),
        "mape": round(float(mape), 2),
    }

    logger.info("\n" + "=" * 40)
    logger.info(f"  MAE  : €{mae:,.0f}/m²")
    logger.info(f"  RMSE : €{rmse:,.0f}/m²")
    logger.info(f"  R²   : {r2:.4f}")
    logger.info(f"  MAPE : {mape:.1f}%")
    logger.info("=" * 40)

    return metrics


if __name__ == "__main__":
    evaluate_model()
