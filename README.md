# CLV Marketing Pipeline

This project implements a complete end-to-end Machine Learning pipeline for Customer Lifetime Value (CLV) prediction. It includes data processing, model training, experiment tracking, model serving, and a user-facing dashboard.

## Project Structure

The project is containerized using Docker and consists of the following services:

*   **Airflow (`airflow-webserver`, `airflow-scheduler`)**: Orchestrates data processing and model training workflows.
*   **MLflow (`mlflow`)**: Tracks experiments, metrics, and models.
*   **API (`api`)**: FastAPI application that serves the trained model and customer data.
*   **Frontend (`frontend`)**: Streamlit application for visualizing customer segments and triggering predictions.
*   **Postgres (`postgres`)**: Metadata database for Airflow.

## Prerequisites

*   Docker
*   Docker Compose

## Getting Started

1.  **Clone the repository** (if you haven't already).
2.  **Start the services**:
    ```bash
    docker-compose up --build
    ```
    This will build the images and start all services.

## Accessing Services

| Service | URL | Credentials (if any) |
| :--- | :--- | :--- |
| **Airflow** | [http://localhost:8080](http://localhost:8080) | `airflow` / `airflow` |
| **MLflow** | [http://localhost:5000](http://localhost:5000) | None |
| **Frontend** | [http://localhost:8501](http://localhost:8501) | None |
| **API** | [http://localhost:8000](http://localhost:8000) | None |
| **API Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | None |

## Workflow

### 1. Data Pipeline (`daily_data_update` DAG)
*   **Schedule**: Runs daily.
*   **Purpose**: ingestion and processing of raw data.
*   **Steps**:
    1.  Loads raw data (`data/Online_Retail.xlsx`).
    2.  Cleans the data.
    3.  Calculates RFM (Recency, Frequency, Monetary) features.
    4.  Performs K-Means clustering for customer segmentation.
    5.  Saves processed data and feature files to the `data/` directory.

### 2. Model Training (`training_pipeline` DAG)
*   **Schedule**: Manual trigger.
*   **Purpose**: Train the CLV prediction model.
*   **Steps**:
    1.  Prepares training data from processed data.
    2.  Trains an XGBoost Regressor model.
    3.  Logs parameters and metrics (RMSE, R2) to **MLflow**.
    4.  Saves the trained model to `models/` for the API to use.

### 3. Model Serving & Usage
*   **API**: Loads the saved models (`clv_model.joblib`, `kmeans_model.joblib`, etc.) from the `models/` directory and exposes endpoints to fetch customer details and predict CLV.
*   **Frontend**:
    *   **Dashboard**: Visualizes customer segments and revenue.
    *   **Prediction**: Allows users to enter a Customer ID to get their predicted CLV and segment.
    *   **Model Training**: Provides a button to trigger the `training_pipeline` in Airflow directly from the UI.
