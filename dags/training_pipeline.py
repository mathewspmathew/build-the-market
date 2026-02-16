from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

DATA_PATH = "/opt/airflow/data"
RAW_FILE = os.path.join(DATA_PATH, "Online_Retail.xlsx")
MODELS_DIR = "/opt/airflow/models"

def run_training():
    train(RAW_FILE, MODELS_DIR)

with DAG(
    'training_pipeline',
    default_args=default_args,
    description='Manually triggered model training pipeline',
    schedule_interval=None, # Manual trigger only
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['ml', 'training'],
) as dag:

    train_task = PythonOperator(
        task_id='train_model_task',
        python_callable=run_training,
    )

    train_task
