import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    # Load dataset PREPROCESSED
    df = pd.read_csv("../MLProject/credit_card_fraud_preprocessed.csv")

    # Split feature & target
    X = df.drop(columns=["IsFraud"])
    y = df["IsFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # MLflow local tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("Credit Card Fraud Detection")

    # Enable autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run():

        # Model (NO TUNING)
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metric
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

        # ===== ARTIFACT 1 =====
        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        # ===== ARTIFACT 2 =====
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")


if __name__ == "__main__":
    train_model()

