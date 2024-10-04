import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Aggregate and transform features
def aggregate_features(df):
    df['Total_Transaction_Amount'] = df.groupby('CustomerId')['Amount'].transform('sum')
    df['Transaction_Count'] = df.groupby('CustomerId')['Amount'].transform('count')
    return df

# Encode categorical variables
def encode_categorical(df):
    encoder = OneHotEncoder()
    categorical_cols = ['ProductCategory', 'ChannelId']  # Add more categorical columns as needed
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded

# Normalize/standardize numerical features
def scale_features(df, numerical_cols):
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

if __name__ == "__main__":
    filepath = "C:\\Users\\HP\\week 6\\Credit-Scoring-Model\\data\\data.csv"
    df = pd.read_csv(filepath)
    df = aggregate_features(df)
    df = encode_categorical(df)
    df = scale_features(df, numerical_cols=['Amount', 'Total_Transaction_Amount'])
    df.to_csv("C:\\Users\\HP\\week 6\\Credit-Scoring-Model\\data\\processed\\processed_data.csv", index=False)
