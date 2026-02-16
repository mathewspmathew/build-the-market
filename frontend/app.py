import streamlit as st
import pandas as pd
import requests
import os
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://localhost:8080")
# Basic Auth for Airflow (default in docker-compose)
AIRFLOW_AUTH = ("airflow", "airflow")

st.set_page_config(page_title="CLV Prediction Dashboard", layout="wide")

st.title("Customer Lifetime Value Prediction Dashboard")

# Sidebar for Navigation
page = st.sidebar.selectbox("Navigation", ["Dashboard", "Prediction", "Model Training"])

DATA_PATH = "/opt/airflow/data"

@st.cache_data
def load_data():
    # Attempt to load data if it exists
    # In docker, we mount the data volume
    try:
        rfm_file = os.path.join(DATA_PATH, "rfm_with_cluster.csv")
        if os.path.exists(rfm_file):
            return pd.read_csv(rfm_file)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

if page == "Dashboard":
    st.header("Customer Segmentation Dashboard")
    
    df = load_data()
    
    if df is not None:
        # Metrics
        st.metric("Total Customers", len(df))
        st.metric("Total Revenue", f"${df['Monetary'].sum():,.2f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Segments Distribution")
            if 'Tags' in df.columns:
                fig, ax = plt.subplots()
                df['Tags'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Segmentation tags not found.")
                
        with col2:
            st.subheader("Revenue by Segment")
            if 'Tags' in df.columns:
                fig, ax = plt.subplots()
                revenue_by_segment = df.groupby('Tags')['Monetary'].sum().sort_values(ascending=False)
                sns.barplot(x=revenue_by_segment.values, y=revenue_by_segment.index, ax=ax)
                st.pyplot(fig)
        
        st.subheader("Cluster Analysis")
        st.dataframe(df.groupby('Tags')[['Recency', 'Frequency', 'Monetary']].mean())
        
    else:
        st.warning("Data not available yet. Please run the data pipeline.")

elif page == "Prediction":
    st.header("CLV Prediction")
    
    customer_id = st.number_input("Enter Customer ID", min_value=1, value=12347)
    
    if st.button("Predict CLV"):
        try:
            # Call API
            response = requests.get(f"{API_URL}/customer/{customer_id}")
            
            if response.status_code == 200:
                data = response.json()
                st.success(f"Prediction Successful for Customer {customer_id}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted CLV (3 Months)", f"${data['predicted_clv']}")
                col2.metric("Segment", data['Segment'])
                col3.metric("Current Cluster", data['Cluster'])
                
                st.subheader("Customer Details")
                st.json(data)
            elif response.status_code == 404:
                st.error("Customer not found.")
            else:
                st.error(f"Error: {response.text}")
                
        except Exception as e:
            st.error(f"Connection error: {e}")

elif page == "Model Training":
    st.header("Model Management")
    
    st.write("Trigger the training pipeline manually. This will train a new XGBoost model and log it to MLflow.")
    
    if st.button("Trigger Training"):
        # Trigger Airflow DAG
        dag_id = "training_pipeline"
        url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns"
        
        try:
            response = requests.post(
                url, 
                json={"conf": {}}, 
                auth=AIRFLOW_AUTH
            )
            
            if response.status_code == 200:
                st.success("Training Pipeline Triggered Successfully!")
                st.json(response.json())
            else:
                st.error(f"Failed to trigger pipeline: {response.status_code} - {response.text}")
                
        except Exception as e:
            st.error(f"Connection error to Airflow: {e}")
            
    st.subheader("MLflow")
    st.markdown(f"[Open MLflow UI]({AIRFLOW_URL.replace('8080', '5000')})")
    st.info("Visit MLflow to track experiments and metrics.")
