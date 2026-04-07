from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "final.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "real_estate_model.pkl"

TARGET_COLUMN = "price"

FEATURE_COLUMNS = [
    "year_sold",
    "property_tax",
    "insurance",
    "beds",
    "baths",
    "sqft",
    "year_built",
    "lot_size",
    "basement",
    "popular",
    "recession",
    "property_age",
    "property_type_Condo",
]