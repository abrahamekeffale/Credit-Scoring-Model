import pandas as pd

def load_data(file_path):
    """Load dataset from a CSV file"""
    return pd.read_csv(file_path)

def feature_engineering(df):
    """Perform any additional feature engineering on the dataset"""
    # Example: Creating new features
    df['Transaction_Ratio'] = df['Total_Transaction_Amount'] / df['Transaction_Count']
    return df
