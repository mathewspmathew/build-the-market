from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import load_data, clean_data
from src.feature_engineering import calculate_features, prepare_training_data, perform_clustering
import pandas as pd

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

DATA_PATH = "/opt/airflow/data"
MODELS_PATH = "/opt/airflow/models"
RAW_FILE = os.path.join(DATA_PATH, "Online_Retail.xlsx")
CLEAN_FILE = os.path.join(DATA_PATH, "cleaned_data.csv")
RFM_FILE = os.path.join(DATA_PATH, "rfm_features.csv")
RFM_CLUSTER_FILE = os.path.join(DATA_PATH, "rfm_with_cluster.csv")

def process_data():
    if not os.path.exists(RAW_FILE):
        print(f"File not found: {RAW_FILE}")
        return

    print(f"Loading data from {RAW_FILE}")
    df = load_data(RAW_FILE)
    
    print("Cleaning data...")
    cleaned_df = clean_data(df)
    
    print(f"Saving cleaned data to {CLEAN_FILE}")
    cleaned_df.to_csv(CLEAN_FILE, index=False)
    
    # Calculate Features (All interactions for segmentation/dashboard)
    # We take the max date in the dataset as the snapshot date for "current state"
    snapshot_date = cleaned_df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    print("Calculating RFM features...")
    features = calculate_features(cleaned_df, snapshot_date)
    
    print(f"Saving RFM features to {RFM_FILE}")
    features.to_csv(RFM_FILE)

    print("Performing Clustering...")
    # Pass models_dir to save the model and scaler for the API to use
    features_with_cluster = perform_clustering(features, n_clusters=4, models_dir=MODELS_PATH)
    
    print(f"Saving RFM with clusters to {RFM_CLUSTER_FILE}")
    features_with_cluster.to_csv(RFM_CLUSTER_FILE)

with DAG(
    'daily_data_update',
    default_args=default_args,
    description='Daily data ingestion and processing',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:

    process_task = PythonOperator(
        task_id='process_data_task',
        python_callable=process_data,
    )

    process_task
