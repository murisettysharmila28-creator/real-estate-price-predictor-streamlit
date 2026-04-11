import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.config import TARGET_COLUMN, FEATURE_COLUMNS, MODEL_DIR, MODEL_PATH
from src.evaluate import evaluate_model
from src.logger import setup_logger
from src.custom_exception import CustomException
from src.validation import validate_model

logger = setup_logger()


def prepare_data(df):
    try:
        x = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]

        logger.info("Real estate data prepared successfully.")
        return x, y

    except Exception as e:
        logger.error("Error occurred while preparing real estate data.", exc_info=True)
        raise CustomException(e, sys)


def split_data(x, y):
    try:
        return train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42
        )

    except Exception as e:
        logger.error("Error occurred during train-test split.", exc_info=True)
        raise CustomException(e, sys)


def train_linear_regression(x_train, y_train):
    try:
        model = LinearRegression()
        model.fit(x_train, y_train)

        logger.info("Linear Regression model trained successfully.")
        return model

    except Exception as e:
        logger.error("Error occurred during Linear Regression training.", exc_info=True)
        raise CustomException(e, sys)


def train_random_forest(x_train, y_train):
    try:
        model = RandomForestRegressor(
            n_estimators=200,
            criterion="absolute_error",
            random_state=42
        )
        model.fit(x_train, y_train)

        logger.info("Random Forest model trained successfully.")
        return model

    except Exception as e:
        logger.error("Error occurred during Random Forest training.", exc_info=True)
        raise CustomException(e, sys)


def save_model(model):
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Model saved at: {MODEL_PATH}")

    except Exception as e:
        logger.error("Error occurred while saving the real estate model.", exc_info=True)
        raise CustomException(e, sys)


def train_and_select_best_model(df):
    try:
        x, y = prepare_data(df)
        x_train, x_test, y_train, y_test = split_data(x, y)

        lr_model = train_linear_regression(x_train, y_train)
        lr_train_mae, lr_test_mae = evaluate_model(
            lr_model, x_train, y_train, x_test, y_test
        )

        rf_model = train_random_forest(x_train, y_train)
        rf_train_mae, rf_test_mae = evaluate_model(
            rf_model, x_train, y_train, x_test, y_test
        )

        if rf_test_mae < lr_test_mae:
            best_model = rf_model
            best_name = "Random Forest"
            best_mae = rf_test_mae
        else:
            best_model = lr_model
            best_name = "Linear Regression"
            best_mae = lr_test_mae

        logger.info(
            f"Best model selected: {best_name} with Test MAE = {best_mae:.2f}"
        )

        validation_results = validate_model(best_model, x, y, cv=5)

        save_model(best_model)

        return best_model, best_name, best_mae, validation_results

    except Exception as e:
        logger.error("Error occurred during real estate training pipeline.", exc_info=True)
        raise CustomException(e, sys)