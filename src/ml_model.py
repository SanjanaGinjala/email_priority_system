import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

MODEL_PATH = "../models/email_priority_model.pkl"
VECTORIZER_PATH = "../models/tfidf_vectorizer.pkl"


def train_and_save_model(df):
    os.makedirs("../models", exist_ok=True)

    # ----- TEXT FEATURES -----
    X_text = df["subject"] + " " + df["body"]

    # ----- LABELS (SAFE FALLBACK) -----
    # If user feedback exists → use it
    # Else → use rule-based priority
    if "priority_updated" in df.columns:
        y = df["priority_updated"].fillna(df["priority_rule_label"])
    else:
        y = df["priority_rule_label"]

    # ----- VECTORIZATION -----
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=500
    )
    X = vectorizer.fit_transform(X_text)

    # ----- MODEL -----
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    model.fit(X, y)

    # ----- SAVE -----
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("✅ ML model and vectorizer saved")


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Train the model first using train_and_save_model(df)"
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_priority_ml(email_row, model, vectorizer):
    text = f"{email_row['subject']} {email_row['body']}"
    X = vectorizer.transform([text])
    return model.predict(X)[0]
