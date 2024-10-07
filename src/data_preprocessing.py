import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def handle_missing_values(df):
    """Fill missing values in the dataset"""
    return df.fillna(df.median())

def encode_categorical_columns(df, categorical_columns):
    """Encode categorical columns using one-hot encoding"""
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_cols = pd.DataFrame(encoder.fit_transform(df[categorical_columns]))
    encoded_cols.columns = encoder.get_feature_names_out(categorical_columns)
    df = df.drop(categorical_columns, axis=1)
    df = pd.concat([df, encoded_cols], axis=1)
    return df

def split_data(df, target):
    """Split the data into training and testing sets"""
    X = df.drop(target, axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)
