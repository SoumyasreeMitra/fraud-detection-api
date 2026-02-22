import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from preprocess import load_and_preprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "features.pkl")


def train():

    df = load_and_preprocess(DATA_PATH)

    X = df.drop("Fraud", axis=1)
    y = df["Fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nROC-AUC:", roc_auc_score(y_test, y_prob))
    print("PR-AUC:", average_precision_score(y_test, y_prob))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(X.columns.tolist(), FEATURE_PATH)

    print("\nModel saved successfully.")
threshold = 0.2

y_pred = (y_prob > threshold).astype(int)

fp = ((y_pred == 1) & (y_test == 0)).sum()
fn = ((y_pred == 0) & (y_test == 1)).sum()

total_loss = (fn * 1000) + (fp * 10)

print("Estimated Financial Loss:", total_loss)

if __name__ == "__main__":
    train()