import pandas as pd
import os

# Get absolute path to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
UPDATE_FILE = os.path.join(OUTPUT_DIR, "updated_emails.csv")
DATA_PATH = os.path.join(BASE_DIR, "data", "emails_day1.csv")
def load_emails(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

    else:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"Default dataset missing. Ensure data/emails_day1.csv is committed."
            )
        df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Uploaded dataset is empty")

    if "email_id" not in df.columns:
        df.insert(0, "email_id", range(1, len(df) + 1))

    return df
