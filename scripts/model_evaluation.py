from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probas)
    report = classification_report(y_test, predictions)
    return auc, report

if __name__ == "__main__":
    # Example usage:
    auc, report = evaluate_model(model, X_test, y_test)  # Pass your model and data here
    print(f"Model AUC: {auc}")
    print("Classification Report:\n", report)
