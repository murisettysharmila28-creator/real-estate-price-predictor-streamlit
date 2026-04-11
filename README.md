# Real Estate Price Prediction (Streamlit App)

## Overview

This project presents an end-to-end machine learning pipeline for predicting real estate prices using structured housing data. The original notebook-based workflow was transformed into a modular, production-oriented Python project with proper logging, error handling, validation, and deployment using Streamlit.

The application enables users to input property attributes and obtain a real-time estimate of property price, making the solution suitable for exploratory analysis and decision support.

Live Application:
https://real-estate-price-predictor-app-sharmila.streamlit.app

---

## Problem Statement

Notebook-based machine learning solutions often lack modularity, reproducibility, and robustness required for real-world deployment. They typically do not incorporate structured validation or consistent pipelines for training and inference.

This project addresses these challenges by:
- converting exploratory code into a modular pipeline
- ensuring consistent feature handling across training and prediction
- incorporating logging and structured error handling
- validating model performance using cross-validation
- deploying the model for real-time inference

---

## Dataset

The dataset contains housing and market-related attributes used to estimate property price.

### Features
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

The dataset was preprocessed to ensure numerical consistency and proper feature selection for regression modelling.

---

## Code Modularization Approach

The project follows a structured design aligned with real-world machine learning engineering practices:

- `data_loader.py` - Handles dataset ingestion with error handling  
- `train.py` - Model training, comparison, and selection  
- `evaluate.py` - Computes regression metrics (MAE)  
- `validation.py` - Performs cross-validation for robustness  
- `predict.py` - Handles inference pipeline  
- `logger.py` - Centralized logging system  
- `custom_exception.py` - Structured exception handling  

This modular design improves maintainability, debugging, and scalability.

---

## Modelling Approach

Two regression models were evaluated:

- Linear Regression  
- Random Forest Regressor  

Random Forest was selected as the final model due to its ability to capture non-linear relationships and interactions between features, which are common in real estate pricing.

Unlike Linear Regression, which assumes linearity, Random Forest leverages ensemble learning to reduce variance and improve predictive performance.

---

## Training Results

Both models were trained and evaluated using Mean Absolute Error (MAE) as the primary metric.

Linear Regression provided a baseline with interpretable but limited performance, while Random Forest achieved lower error and better captured complex feature relationships.

---

## Test Results

- Random Forest Test MAE: 47,323.66  

The model achieved a relatively low error on the hold-out test set, indicating good predictive capability.

---

## Validation Results

To assess model reliability beyond a single train-test split, 5-fold cross-validation was performed using Mean Absolute Error.

### Cross-Validation MAE Scores

- 36,031.57  
- 38,657.02  
- 42,692.24  
- 53,089.13  
- 71,276.22  

### Average Cross-Validation MAE

- 48,349.23  

### Standard Deviation (Approximate Insight)

The variation across folds indicates some sensitivity to data splits, particularly due to potential outliers or distribution differences in housing data.

### Interpretation

The average cross-validation MAE (48,349.23) is very close to the hold-out test MAE (47,323.66), which suggests that the model generalizes reasonably well and is not overfitting to a single split.

The variability across folds highlights that real estate data can be heterogeneous, but the model still maintains overall stability.

---

## Streamlit Application

The Streamlit application provides an interactive interface for real-time price prediction.

Features:
- Input housing attributes  
- Real-time price estimation  
- Clean and user-friendly interface  

The app uses the trained model to generate predictions consistently with the training pipeline.

---

## Logging and Error Handling

The project includes production-level logging and exception handling:

- Logs are stored in `logs/app.log`  
- Tracks:
  - data loading  
  - model training  
  - evaluation and validation  
  - prediction flow  
  - runtime errors with stack traces  

Custom exception handling ensures:
- traceable errors  
- improved debugging  
- robust execution of the pipeline  

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
├── logs/
│   └── app.log
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   ├── validation.py
│   ├── predict.py
│   ├── logger.py
│   └── custom_exception.py
│
└── notebooks/
    └── Real_Estate.ipynb
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/murisettysharmila28-creator/real-estate-price-predictor-streamlit
cd real-estate-price-predictor-streamlit
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Train the model:

```bash
python main.py
```

Run the Streamlit app:

```bash
python -m streamlit run app.py
```

---

## Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Joblib  

---

## Key Findings

- Random Forest outperformed Linear Regression due to its ability to model non-linear relationships  
- Cross-validation confirmed that model performance is stable across different data splits  
- MAE is an effective metric for interpreting real estate prediction errors in monetary terms  
- Feature interactions play a significant role in price prediction  

---

## Limitations

- Model performance depends on dataset quality and coverage  
- Location-specific features are limited  
- Predictions are estimates and may not reflect real market fluctuations  
- Some variability exists across different validation folds  

---

## Challenges

- Selecting appropriate regression metrics for evaluation  
- Handling variability in housing data  
- Ensuring model stability across splits  
- Designing a modular and production-ready pipeline  

---

## Learning Outcomes

- Built a complete regression pipeline from scratch  
- Applied cross-validation for robust evaluation  
- Implemented logging and structured error handling  
- Compared linear and ensemble models effectively  
- Developed and deployed an interactive ML application  

---

## Future Enhancements

- Hyperparameter tuning for Random Forest  
- Incorporation of location-based features  
- Advanced models such as Gradient Boosting or XGBoost  
- Improved feature engineering  
- Enhanced UI with visual insights  

---

## Author

Sharmila Murisetty  
Data Analyst / Business Intelligence Developer