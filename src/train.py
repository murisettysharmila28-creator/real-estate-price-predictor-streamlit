import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.config import TARGET_COLUMN, FEATURE_COLUMNS, MODEL_DIR, MODEL_PATH
from src.evaluate import evaluate_model
from src.logger import setup_logger

logger = setup_logger()


def prepare_data(df):
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return x, y


def split_data(x, y):
    return train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )


def train_linear_regression(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    logger.info("Linear Regression model trained successfully.")
    return model


def train_random_forest(x_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        criterion="absolute_error",
        random_state=42
    )
    model.fit(x_train, y_train)
    logger.info("Random Forest model trained successfully.")
    return model


def save_model(model):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved at: {MODEL_PATH}")


def train_and_select_best_model(df):
    x, y = prepare_data(df)
    x_train, x_test, y_train, y_test = split_data(x, y)

    lr_model = train_linear_regression(x_train, y_train)
    lr_train_mae, lr_test_mae = evaluate_model(lr_model, x_train, y_train, x_test, y_test)

    rf_model = train_random_forest(x_train, y_train)
    rf_train_mae, rf_test_mae = evaluate_model(rf_model, x_train, y_train, x_test, y_test)

    if rf_test_mae < lr_test_mae:
        best_model = rf_model
        best_name = "Random Forest"
        best_mae = rf_test_mae
    else:
        best_model = lr_model
        best_name = "Linear Regression"
        best_mae = lr_test_mae

    logger.info(f"Best model selected: {best_name} with Test MAE = {best_mae:.2f}")
    save_model(best_model)

    return best_model, best_name, best_mae