import pandas as pd
import os

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from an Excel or CSV file.
    """
    if filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)
    elif filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    else:
        raise ValueError("Unsupported file format. Please use .xlsx or .csv")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the input dataframe:
    - Drop rows with missing CustomerID
    - Remove cancelled orders (InvoiceNo starts with 'C')
    - Convert InvoiceDate to datetime
    - Remove rows with non-positive Quantity or UnitPrice
    - Calculate Revenue
    """
    # Drop missing CustomerID
    df = df.dropna(subset=['CustomerID'])
    
    # Remove cancelled orders
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    
    # Convert to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # positive quantity and unit price
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    # Calculate Revenue
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    
    return df
