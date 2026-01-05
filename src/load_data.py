import pandas as pd
import os

DATA_PATH = "../data/emails_day1.csv"

def load_emails(uploaded_file=None):
    """
    Loads emails either from uploaded CSV or default dataset
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)

    # Ensure email_id exists
    if "email_id" not in df.columns:
        df.insert(0, "email_id", range(1, len(df) + 1))

    return df
