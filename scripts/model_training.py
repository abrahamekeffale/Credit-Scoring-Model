import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load processed data
def load_data(filepath):
    return pd.read_csv(filepath)

# Train Random Forest model
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Train Logistic Regression model
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    df = load_data("C:\\Users\\HP\\week 6\\Credit-Scoring-Model\\data\\processed\\processed_data.csv")
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='default_label'), df['default_label'], test_size=0.2)
    
    rf_model = train_random_forest(X_train, y_train)
    lr_model = train_logistic_regression(X_train, y_train)
    
    rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    lr_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])
    
    print(f"Random Forest AUC: {rf_auc}")
    print(f"Logistic Regression AUC: {lr_auc}")
