import pickle
import pandas as pd
from src.config import MODEL_PATH, FEATURE_COLUMNS
from src.logger import setup_logger

logger = setup_logger()


def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        logger.error("Model file not found. Train the model first.")
        raise


def predict_price(input_data: dict):
    model = load_model()
    input_df = pd.DataFrame([input_data])[FEATURE_COLUMNS]
    prediction = model.predict(input_df)[0]
    logger.info(f"Prediction generated: {prediction}")
    return prediction