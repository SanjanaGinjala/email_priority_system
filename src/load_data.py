import pandas as pd
import os
import io

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "emails_day1.csv")

def load_emails(uploaded_file=None):
    if uploaded_file is not None:
        # ✅ Streamlit uploaded file handling
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(stringio)
    else:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                "Default dataset missing. Ensure data/emails_day1.csv exists."
            )
        df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Dataset is empty")

    if "email_id" not in df.columns:
        df.insert(0, "email_id", range(1, len(df) + 1))

    return df