import pandas as pd
import os

# Get absolute path to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
UPDATE_FILE = os.path.join(OUTPUT_DIR, "updated_emails.csv")
DATA_PATH = os.path.join(BASE_DIR, "data", "emails_day1.csv")
def load_emails(uploaded_file=None):
    # Case 1: Uploaded dataset
    if uploaded_file is not None:
        uploaded_file.seek(0)  # 🔥 RESET FILE POINTER
        df = pd.read_csv(uploaded_file)

    # Case 2: Default dataset
    else:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)

    # Ensure email_id exists
    if "email_id" not in df.columns:
        df.insert(0, "email_id", range(1, len(df) + 1))
    if df.empty:
        raise ValueError("Uploaded dataset is empty")

    return df
