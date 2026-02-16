import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

def calculate_features(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """
    Calculate RFM and other behavioral features for each customer.
    
    Args:
        df: Cleaned transaction DataFrame containing 'CustomerID', 'InvoiceDate', 'InvoiceNo', 'Revenue'.
        snapshot_date: The date relative to which recency and age are calculated.
        
    Returns:
        pd.DataFrame: Feature dataframe with CustomerID as index.
    """
    
    # Aggregations
    # Recency: days since last purchase
    # Frequency: count of unique invoices
    # Monetary: sum of revenue
    # AvgRevenue: mean of revenue per transaction (line item or invoice? Notebook does grouping by CustomerID on raw data, so it's mean of line item revenue... wait, let's verify notebook)
    # The notebook does: extra = df.groupby("CustomerID").agg({"Revenue": ["mean", "std"], ...})
    # So it's mean of line item revenue.
    
    # Let's perform all aggregations in one go to be efficient
    features = df.groupby('CustomerID').agg({
        'InvoiceDate': [
            lambda x: (snapshot_date - x.max()).days, # Recency
            'min' # FirstPurchase for Age
        ],
        'InvoiceNo': 'nunique', # Frequency
        'Revenue': ['sum', 'mean', 'std'] # Monetary, AvgRevenue, StdRevenue
    })
    
    # Flatten MultiIndex columns
    features.columns = [
        'Recency', 
        'FirstPurchase', 
        'Frequency', 
        'Monetary', 
        'AvgRevenue', 
        'StdRevenue'
    ]
    
    # Calculate Customer Age
    features['CustomerAgeDays'] = (snapshot_date - features['FirstPurchase']).dt.days
    
    # Drop FirstPurchase as it's a date and we have Age
    features.drop(columns=['FirstPurchase'], inplace=True)
    
    # Fill NaN values (StdRevenue can be NaN if only 1 transaction)
    features.fillna(0, inplace=True)
    
    return features

def prepare_training_data(df: pd.DataFrame, cutoff_date: pd.Timestamp, prediction_window_days: int = 90):
    """
    Split data into past (features) and future (target) based on cutoff_date.
    
    Args:
        df: Full cleaned DataFrame.
        cutoff_date: The split point.
        prediction_window_days: How many days into the future to predict revenue for (default 90 days = 3 months).
        
    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Target (Log-transformed future revenue)
    """
    # Split data
    train_df = df[df['InvoiceDate'] <= cutoff_date]
    
    # Future data for target (only strictly after cutoff)
    future_start = cutoff_date
    future_end = cutoff_date + pd.Timedelta(days=prediction_window_days)
    future_df = df[(df['InvoiceDate'] > future_start) & (df['InvoiceDate'] <= future_end)]
    
    # Snapshot date for features is immediately after cutoff
    snapshot_date = cutoff_date + pd.Timedelta(days=1)
    
    # Calculate Features (X)
    X = calculate_features(train_df, snapshot_date)
    
    # Calculate Target (y)
    future_revenue = future_df.groupby('CustomerID')['Revenue'].sum()
    
    # Align X and y (keep all customers in X, fill missing y with 0)
    # This captures churners (those who bought in past but 0 in future)
    
    # Create a DataFrame for y to merge
    y_df = pd.DataFrame(future_revenue)
    y_df.columns = ['target_revenue']
    
    # Merge
    merged = X.merge(y_df, on='CustomerID', how='left')
    merged['target_revenue'].fillna(0, inplace=True)
    
    # Log transform target as per notebook
    # y = np.log1p(y_raw)
    y = np.log1p(merged['target_revenue'])
    
    # Features
    X_final = merged.drop(columns=['target_revenue'])
    
    return X_final, y

from typing import Optional

def perform_clustering(features: pd.DataFrame, n_clusters: int = 4, models_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Perform KMeans clustering on RFM features and assign human-readable tags.
    
    Args:
        features: DataFrame with 'Recency', 'Frequency', 'Monetary' columns.
        n_clusters: Number of clusters (default 4 based on notebook).
        models_dir: Directory to save the trained model and scaler (optional).
    
    Returns:
        pd.DataFrame: DataFrame with additional 'Cluster' and 'Tags' columns.
    """
    df_rfm = features[['Recency', 'Frequency', 'Monetary']].copy()
    
    # Scale data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_rfm)
    
    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)
    
    # Helper to assign tags based on cluster centroids
    # calculated centroids mean values
    df_rfm['Cluster'] = clusters
    cluster_means = df_rfm.groupby('Cluster').mean()
    
    # Logic to map clusters to tags:
    # "Vip list": Highest Monetary
    # "One time customers": Lowest Frequency (approx 1)
    # "Recent customers": Lowest Recency (most recent)
    # "Loyal customers": Remaining or High Frequency but not VIP
    
    # Initialize tagging
    tags = {}
    available_clusters = set(range(n_clusters))
    
    # 1. VIP: Max Monetary
    vip_cluster = cluster_means['Monetary'].idxmax()
    tags[vip_cluster] = "Vip list"
    available_clusters.remove(vip_cluster)
    
    # 2. One time: Lowest Frequency (if available)
    if available_clusters:
        one_time_cluster = cluster_means.loc[list(available_clusters), 'Frequency'].idxmin()
        tags[one_time_cluster] = "One time customers"
        available_clusters.remove(one_time_cluster)
        
    # 3. Recent: Lowest Recency (if available)
    if available_clusters:
        recent_cluster = cluster_means.loc[list(available_clusters), 'Recency'].idxmin()
        tags[recent_cluster] = "Recent customers"
        available_clusters.remove(recent_cluster)
        
    # 4. Loyal: Whatever is left (usually high frequency/monetary but not top)
    for c in available_clusters:
        tags[c] = "Loyal customers"
        
    # Assign tags
    features['Cluster'] = clusters
    features['Tags'] = features['Cluster'].map(tags)
    
    # Save models if directory provided
    if models_dir:
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        joblib.dump(kmeans, os.path.join(models_dir, "kmeans_model.joblib"))
        joblib.dump(scaler, os.path.join(models_dir, "rfm_scaler.joblib"))
    
    return features
