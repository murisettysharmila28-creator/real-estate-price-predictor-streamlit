import sys
from sklearn.metrics import mean_absolute_error

from src.logger import setup_logger
from src.custom_exception import CustomException

logger = setup_logger()


def evaluate_model(model, x_train, y_train, x_test, y_test):
    try:
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        logger.info(f"Train MAE: {train_mae:.2f}")
        logger.info(f"Test MAE: {test_mae:.2f}")

        return train_mae, test_mae

    except Exception as e:
        logger.error("Error occurred during real estate model evaluation.", exc_info=True)
        raise CustomException(e, sys)