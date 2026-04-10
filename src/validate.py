from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.logger import setup_logger

logger = setup_logger()


def run_cross_validation(df):
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    model = RandomForestRegressor(
        n_estimators=200,
        criterion="absolute_error",
        random_state=42
    )

    cv_scores = cross_val_score(
        model,
        x,
        y,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    cv_mae_scores = -cv_scores
    avg_cv_mae = np.mean(cv_mae_scores)

    logger.info(f"Cross-validation MAE scores: {cv_mae_scores}")
    logger.info(f"Average cross-validation MAE: {avg_cv_mae:.2f}")

    return cv_mae_scores, avg_cv_mae