import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Aggregate and transform features
def aggregate_features(df):
    """
    Aggregate total transaction amount and transaction count per customer.
    """
    df['Total_Transaction_Amount'] = df.groupby('CustomerId')['Amount'].transform('sum')
    df['Transaction_Count'] = df.groupby('CustomerId')['Amount'].transform('count')
    return df

# Encode categorical variables
def encode_categorical(df):
    """
    Encode categorical variables using one-hot encoding.
    """
    categorical_cols = ['ProductCategory', 'ChannelId']  # Add more categorical columns as needed
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded

# Normalize/standardize numerical features
def scale_features(df, numerical_cols):
    """
    Scale numerical columns using StandardScaler.
    """
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

if __name__ == "__main__":
    # Load the raw data
    filepath = "C:\\Users\\HP\\week 6\\Credit-Scoring-Model\\data\\data.csv"
    df = pd.read_csv(filepath)
    
    # Apply feature engineering functions
    df = aggregate_features(df)
    df = encode_categorical(df)
    df = scale_features(df, numerical_cols=['Amount', 'Total_Transaction_Amount'])
    
    # Save the processed data to a new CSV file
    output_filepath = "C:\\Users\\HP\\week 6\\Credit-Scoring-Model\\data\\processed_data.csv"
    df.to_csv(output_filepath, index=False)
    
    print(f"Processed data saved to {output_filepath}")
