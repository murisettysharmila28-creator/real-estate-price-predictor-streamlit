# Real Estate Price Prediction

## Live App

https://real-estate-price-predictor-app-sharmila.streamlit.app

---

## Overview

This project is an end-to-end machine learning application for predicting real estate prices using structured housing data. The notebook-based solution was modularized into a reusable Python project and deployed as an interactive Streamlit web application.

The application allows users to enter housing-related features and receive a predicted property price in real time.

---

## Objective

The goal of this project was to:
- convert notebook-based machine learning code into a modular Python project
- compare multiple regression models
- evaluate model performance using error metrics
- add additional validation for model stability
- deploy the final model through Streamlit

---

## Dataset

The dataset contains housing-related and market-related features used to estimate property price.

### Features used
- year_sold
- property_tax
- insurance
- beds
- baths
- sqft
- year_built
- lot_size
- basement
- popular
- recession
- property_age
- property_type_Condo

### Target
- price

---

## Models Used

The following regression models were tested:
- Linear Regression
- Random Forest Regressor

### Selected Model
- **Random Forest Regressor**

The Random Forest model was selected because it achieved the better test performance.

---

## Evaluation and Validation

### Hold-out Test Result
- **Best Test MAE:** 47,323.66

### Additional Validation
To assess model stability beyond a single train-test split, 5-fold cross-validation was performed using Mean Absolute Error (MAE).

#### Cross-Validation MAE Scores
- 36,031.57
- 38,657.02
- 42,692.24
- 53,089.13
- 71,276.22

#### Average Cross-Validation MAE
- **48,349.23**

### Interpretation
The average cross-validation MAE was close to the hold-out test MAE, which suggests that the Random Forest model performed reasonably consistently across different data splits.

---

## Project Features

- Modular code structure
- Separate training, evaluation, validation, and prediction modules
- Logging and error handling
- Model comparison using MAE
- Additional 5-fold cross-validation for model validation
- Interactive Streamlit interface for price prediction
- Deployed web application for real-time inference

---

## Project Structure

```bash
real-estate-price-predictor-streamlit/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
│
├── data/
│   └── final.csv
│
├── model/
│   └── real_estate_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   ├── validate.py
│   ├── predict.py
│   └── logger.py
│
└── notebooks/
    └── Real_Estate.ipynb

```

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/real-estate-price-predictor-streamlit.git
cd real-estate-price-predictor-streamlit

### 2. Install dependencies
pip install -r requirements.txt

### 3. Train the model
python main.py

### 4. Run the Streamlit app
python -m streamlit run app.py
```
## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Key Learnings

- Converting notebook code into a modular ML project
- Comparing regression models using MAE
- Using cross-validation to assess model stability
- Saving model artifacts for deployment
- Building an interactive prediction app using Streamlit

## Limitations

- The model depends on the quality and scope of the dataset
- Important real-world location factors may not be fully captured
- Predictions should be interpreted as estimates rather than exact market values

## Future Improvements

- Hyperparameter tuning for Random Forest
- Additional feature engineering
- Incorporation of location-based variables
- More robust validation using grid search or randomized search
- Improved user interface and richer visualizations

## Author

Sharmila Murisetty - Data Analyst / BI Developer