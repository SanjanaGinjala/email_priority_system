import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from load_data import load_emails
from priority_engine import calculate_priority
from nlp_keywords import extract_important_keywords
from ml_model import load_model, predict_priority_ml


# -------------------------
# Helper
# -------------------------
def assign_priority_label(score):
    if score >= 10:
        return "HIGH"
    elif score >= 6:
        return "MEDIUM"
    else:
        return "LOW"


# -------------------------
# Load Data
# -------------------------
def evaluate_models():
    print("\n📊 Evaluating Email Priority Models...\n")

    df = load_emails()
    keywords = extract_important_keywords(df, top_n=20)

    # Rule-based prediction
    df["rule_score"] = df.apply(
        lambda row: calculate_priority(row, dynamic_keywords=keywords),
        axis=1
    )
    df["rule_prediction"] = df["rule_score"].apply(assign_priority_label)

    # ML-based prediction
    ml_model, vectorizer = load_model()
    df["ml_prediction"] = df.apply(
        lambda row: predict_priority_ml(row, ml_model, vectorizer),
        axis=1
    )

    # Ground truth (if available)
    if "priority" not in df.columns:
        print("⚠️ No ground truth priority column found.")
        print("Evaluation skipped.")
        return

    y_true = df["priority"]

    # -------------------------
    # Rule-based Evaluation
    # -------------------------
    print("🔹 Rule-Based Model Evaluation")
    print("Accuracy:",
          accuracy_score(y_true, df["rule_prediction"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, df["rule_prediction"]))
    print("Classification Report:")
    print(classification_report(y_true, df["rule_prediction"]))

    # -------------------------
    # ML-based Evaluation
    # -------------------------
    print("\n🔹 ML-Based Model Evaluation")
    print("Accuracy:",
          accuracy_score(y_true, df["ml_prediction"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, df["ml_prediction"]))
    print("Classification Report:")
    print(classification_report(y_true, df["ml_prediction"]))


# -------------------------
# Run directly
# -------------------------
if __name__ == "__main__":
    evaluate_models()
