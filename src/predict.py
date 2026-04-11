import sys
import pickle
import pandas as pd

from src.config import MODEL_PATH, FEATURE_COLUMNS
from src.logger import setup_logger
from src.custom_exception import CustomException

logger = setup_logger()


def load_prediction_artifacts():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        logger.info("Prediction artifacts loaded successfully.")
        return model

    except Exception as e:
        logger.error("Error occurred while loading real estate model artifact.", exc_info=True)
        raise CustomException(e, sys)


def preprocess_input(input_data: dict):
    try:
        input_df = pd.DataFrame([input_data])
        input_df = input_df[FEATURE_COLUMNS]

        logger.info("Real estate input preprocessing completed successfully.")
        return input_df

    except Exception as e:
        logger.error("Error occurred during real estate input preprocessing.", exc_info=True)
        raise CustomException(e, sys)


def predict_price(input_data: dict):
    try:
        model = load_prediction_artifacts()
        input_df = preprocess_input(input_data)

        prediction = model.predict(input_df)[0]

        logger.info(f"Prediction generated successfully: {prediction}")
        return float(prediction)

    except Exception as e:
        logger.error("Error occurred during real estate price prediction.", exc_info=True)
        raise CustomException(e, sys)