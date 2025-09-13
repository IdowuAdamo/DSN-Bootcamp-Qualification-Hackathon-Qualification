Vehicle Price Prediction Pipeline
Overview
This project implements a machine learning pipeline for predicting vehicle prices using ensemble methods (XGBoost, LightGBM, CatBoost) with cross-validation and hyperparameter tuning. The pipeline preprocesses a dataset with features like model_year, milage, and categorical columns (e.g., brand, model), trains multiple models, and blends their predictions to generate a final submission. The project is designed for a regression task, optimized for RMSE (Root Mean Squared Error), and supports categorical features natively.
Features

Data Preprocessing: Handles missing values, categorical features, and feature engineering.
Model Training: Uses XGBoost, LightGBM, and CatBoost with 5-fold cross-validation.
Hyperparameter Tuning: Optimizes model parameters using GridSearchCV or Optuna.
Ensemble Blending: Combines predictions from multiple models and optional external submissions (sub_1, sub_2).
Logging and Persistence: Logs training progress and saves models for reproducibility.

Requirements

Python 3.9+
Libraries:numpy
pandas
scikit-learn
xgboost==3.0.5
lightgbm
catboost
optuna
joblib
matplotlib
torch


Optional: CUDA Toolkit and cuDNN for GPU support (XGBoost and CatBoost).

Installation

Clone the Repository:
git clone <repository-url>
cd car-price-prediction


Create a Virtual Environment (recommended):
conda create -n vehicle_price_env python=3.9
conda activate vehicle_price_env


Install Dependencies:
pip install -r requirements.txt

Or install manually:
pip install numpy pandas scikit-learn xgboost==3.0.5 lightgbm catboost optuna joblib matplotlib torch





GPU Setup (optional):

Install CUDA Toolkit and cuDNN (see XGBoost GPU and CatBoost GPU).
Verify GPU availability:import torch
print(torch.cuda.is_available())





Dataset
The pipeline expects:

train.csv: Training data with columns like id, brand, model, model_year, milage, price (target).
test.csv: Test data with the same features (excluding price).
used_cars.csv: Original dataset
sample_submission.csv: Template for submission with id and price columns.

Place these files in the project root or update file paths in the code.
Usage

Preprocess Data:Run the preprocessing script or notebook cell to load and prepare data:

Key cells:
Cell 1: Defines cross-validation function for XGBoost (cross_validate_model_x).
Cell 2: Trains XGBoost with optimized parameters.
Cell 7: Blends predictions (XGBoost, LightGBM, CatBoost, optional sub_1/sub_2) and generates submission.csv.

Generate Submission:

The pipeline outputs submission.csv with id and price columns.
Example:sample_sub = pd.DataFrame({'id': test['id'], 'price': test_predsx})
sample_sub.to_csv('submission_xgb.csv', index=False)



Cell 7 blends predictions:sample_sub['price'] = 0.2 * test_preds_xgb + 0.1 * test_preds_cat + 0.7 * test_preds_lgbm
sample_sub.to_csv('submission_ensemble.csv', index=False)
Contact
For issues or contributions, please open a pull request or contact the project maintainer.
