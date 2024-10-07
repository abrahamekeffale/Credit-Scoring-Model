import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Create Aggregate Features
def create_aggregate_features(df):
    df['TotalTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('sum')
    df['AvgTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('mean')
    df['TransactionCount'] = df.groupby('CustomerId')['Amount'].transform('count')
    df['StdTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('std')
    return df

# 2. Extract DateTime Features
def extract_datetime_features(df, datetime_column='TransactionStartTime'):
    df['TransactionHour'] = df[datetime_column].dt.hour
    df['TransactionDay'] = df[datetime_column].dt.day
    df['TransactionMonth'] = df[datetime_column].dt.month
    df['TransactionYear'] = df[datetime_column].dt.year
    return df

# 3. Encode Categorical Variables
def encode_categorical(df):
    # Identify categorical columns (excluding 'Amount' and 'Value')
    categorical_cols = df.columns.difference(['Amount', 'Value'])
    
    # Apply OneHotEncoding to all categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df_encoded

# 4 Handle missing values in the dataset
def handle_missing_values(df):
    # Fill missing values for numerical columns with the median
    num_cols = ['Amount', 'Value']
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Fill missing values for categorical columns with the mode
    cat_cols = df.columns.difference(num_cols)
    df[cat_cols] = df[cat_cols].apply(lambda col: col.fillna(col.mode()[0]))
    
    return df
# 5. Normalize/Standardize Numerical Features
def scale_features(df, method='min-max'):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if method == 'min-max':
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df


