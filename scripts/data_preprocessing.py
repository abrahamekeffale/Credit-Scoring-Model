import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys

def load_data(data_path):
    """Load dataset from the given path."""
    df = pd.read_csv(r'C:\Users\HP\week 6\Credit-Scoring-Model\data\data.csv')
    return df

def handle_missing_values(df):
    """Handle missing values by filling with median."""
    df.fillna(df.median(), inplace=True)
    return df

def split_data(df, target_col, test_size=0.2):
    """Split the dataset into train and test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=42)

def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
