import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib
import argparse
import os

import sys

# Add src to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import load_data, clean_data
from src.feature_engineering import prepare_training_data

def train(data_path, models_dir):
    """
    Train the CLV model using XGBoost and log to MLflow.
    """
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    df = clean_data(df)
    
    # Determine cutoff date: 3 months before the last date in dataset
    # This allows us to have 3 months of 'future' data for target calculation
    max_date = df['InvoiceDate'].max()
    cutoff_date = max_date - pd.DateOffset(months=3)
    
    print(f"Dataset Date Range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    print(f"Training Cutoff Date: {cutoff_date}")
    
    # Feature Engineering
    print("Generating features and target...")
    X, y = prepare_training_data(df, cutoff_date, prediction_window_days=90)
    
    print(f"Feature set shape: {X.shape}")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # MLflow tracking
    mlflow.set_experiment("CLV_Prediction_Training")
    
    with mlflow.start_run():
        # Hyperparameters (from notebook)
        params = {
            "n_estimators": 600,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "objective": "reg:squarederror"
        }
        
        mlflow.log_params(params)
        
        # Train model
        print("Training XGBRegressor...")
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"Validation RMSE: {rmse}")
        print(f"Validation R2: {r2}")
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # Log model
        mlflow.xgboost.log_model(model, "clv_xgb_model")
        
        # Save locally as well
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        local_model_path = os.path.join(models_dir, "clv_xgb_model.json")
        model.save_model(local_model_path)
        print(f"Model saved locally to {local_model_path}")
        
        # Save as joblib for API compatibility
        joblib_path = os.path.join(models_dir, "clv_model.joblib")
        joblib.dump(model, joblib_path)
        print(f"Model saved for API to {joblib_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLV Model")
    parser.add_argument("--data_path", type=str, default="data/Online_Retail.xlsx", help="Path to raw data file")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save trained models")
    
    args = parser.parse_args()
    
    train(args.data_path, args.models_dir)
