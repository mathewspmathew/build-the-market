from fastapi import FastAPI, HTTPException
import joblib
from pathlib import Path
import pandas as pd



app = FastAPI()

BASE_PATH = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_PATH / "models"
DATA_PATH = BASE_PATH / "data"

clv_model = joblib.load(MODEL_PATH/"clv_model.joblib")
kmeans_model = joblib.load(MODEL_PATH/"kmeans_model.joblib")
rfm_scaler = joblib.load(MODEL_PATH/"rfm_scaler.joblib")

rfm_data = pd.read_csv(DATA_PATH/"rfm_with_cluster.csv")

@app.get("/")
def home():
    return {"message":"We do customer segmented marketing here !"}

@app.get("/customer/{customerid}")
def getdetails(customerid: int):
    row = rfm_data[rfm_data["CustomerID"] == customerid]

    if row.empty:
        raise HTTPException(status_code=404,detail="customer not found")

    # result = row.iloc[0].to_dict()

    features = row[["Recency","Frequency","Monetary"]]

    # print(row["Cluster"])
    # cluster = int(row["Cluster"].values[0])
    # print(row["Cluster"])
    clv_pred = clv_model.predict(features)[0]
    # print(clv_pred)


    return {
        "CustomerID" : customerid,
        "Recency": int(row["Recency"].values[0]),
        "Frequency": int(row["Frequency"].values[0]),
        "Monetary": int(row["Monetary"].values[0]),
        "Cluster" : int(row["Cluster"].values[0]),
        "Segment" : row["Tags"],
        "predicted_clv" : round(clv_pred,3)
    }
