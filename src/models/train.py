"""
Entraînement des modèles ML (XGBoost, LightGBM, RandomForest).
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from src.data.split import split_dataset, get_feature_columns
from src.utils.config import MODELS_DIR, settings, save_metadata
from src.utils.logging import logger

# Configurations des modèles disponibles
MODEL_CONFIGS: dict[str, object] = {
    "xgboost": XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=settings.random_seed,
        n_jobs=-1,
    ),
    "lightgbm": LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=settings.random_seed,
        n_jobs=-1,
    ),
    "random_forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=settings.random_seed,
        n_jobs=-1,
    ),
}


def train_model(model_name: str | None = None) -> None:
    """
    Entraîner un modèle ML sur le dataset DVF Angers.

    Args:
        model_name: Nom du modèle (xgboost | lightgbm | random_forest)
                    Défaut : valeur de config DEFAULT_MODEL
    """
    model_name = model_name or settings.default_model

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Modèle '{model_name}' inconnu. Choix : {list(MODEL_CONFIGS.keys())}")

    logger.info(f"Entraînement modèle : {model_name}")
    X_train, X_test, y_train, y_test = split_dataset()

    model = MODEL_CONFIGS[model_name]
    model.fit(X_train, y_train)

    # Évaluation sur le jeu de test
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logger.info(f"MAE  : €{mae:,.0f}/m²")
    logger.info(f"RMSE : €{rmse:,.0f}/m²")
    logger.info(f"R²   : {r2:.3f}")

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importance = (
            pd.DataFrame(
                {"feature": X_train.columns, "importance": model.feature_importances_}
            )
            .sort_values("importance", ascending=False)
        )
        logger.info("\nFeature importance :\n" + importance.to_string(index=False))

    # Sauvegarde
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"model_{model_name}_v1.pkl"
    joblib.dump(model, model_path)
    joblib.dump(list(X_train.columns), MODELS_DIR / "feature_columns.pkl")
    logger.info(f"✓ Modèle sauvegardé : {model_path}")

    save_metadata({
        "model_name": model_name,
        "model_file": model_path.name,
        "features": list(X_train.columns),
        "metrics": {"mae": round(float(mae), 2), "rmse": round(float(rmse), 2), "r2": round(float(r2), 4)},
        "train_size": len(X_train),
        "test_size": len(X_test),
    })


if __name__ == "__main__":
    train_model()
