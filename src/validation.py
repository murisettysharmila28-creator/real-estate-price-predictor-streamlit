import sys
from sklearn.model_selection import cross_val_score

from src.logger import setup_logger
from src.custom_exception import CustomException

logger = setup_logger()


def validate_model(model, x, y, cv=5):
    try:
        cv_scores = -cross_val_score(
            model,
            x,
            y,
            cv=cv,
            scoring="neg_mean_absolute_error"
        )

        validation_results = {
            "cv_mae_scores": [round(float(score), 2) for score in cv_scores],
            "cv_mean_mae": round(float(cv_scores.mean()), 2),
            "cv_std_mae": round(float(cv_scores.std()), 2),
        }

        logger.info(f"Cross-validation MAE scores: {validation_results['cv_mae_scores']}")
        logger.info(f"Average CV MAE: {validation_results['cv_mean_mae']}")
        logger.info(f"CV MAE Std Dev: {validation_results['cv_std_mae']}")

        return validation_results

    except Exception as e:
        logger.error("Error occurred during real estate model validation.", exc_info=True)
        raise CustomException(e, sys)