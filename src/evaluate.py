from sklearn.metrics import mean_absolute_error
from src.logger import setup_logger

logger = setup_logger()


def evaluate_model(model, x_train, y_train, x_test, y_test):
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    logger.info(f"Train MAE: {train_mae:.2f}")
    logger.info(f"Test MAE: {test_mae:.2f}")

    return train_mae, test_mae