import streamlit as st
import pandas as pd
import os

from load_data import load_emails
from priority_engine import calculate_priority
from nlp_keywords import extract_important_keywords
from ml_model import load_model, predict_priority_ml, train_and_save_model

# --------------------------------------------------
# Page Config (MUST be first Streamlit command)
# --------------------------------------------------
st.set_page_config(page_title="Email Priority Dashboard", layout="wide")
st.title("📧 Email Priority Dashboard")

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# --------------------------------------------------
# FIRST SCREEN — Dataset Selection
# --------------------------------------------------
if not st.session_state.data_loaded:

    st.markdown("### 📤 Upload Your Email Dataset (CSV)")

    uploaded_file = st.file_uploader(
        "Upload a CSV file with email data",
        type=["csv"]
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗂️ Use Default Dataset"):
            st.session_state.uploaded_file = None
            st.session_state.data_loaded = True
            st.rerun()
    with col1:
        if uploaded_file is not None:
            if st.button("📂 Use Uploaded Dataset"):
                st.session_state.uploaded_file = uploaded_file
                st.session_state.data_loaded = True
                st.rerun()

    st.stop()  # ⛔ Stop app here until dataset is chosen

# --------------------------------------------------
# Load Data (AFTER button click)
# --------------------------------------------------
df = load_emails(st.session_state.uploaded_file)
top_keywords = extract_important_keywords(df, top_n=20)
st.caption(f"📊 Loaded {len(df)} emails")
if "priority_rule_label" not in df.columns:
    df["priority_rule_label"] = df.get("priority", 0)
try:
    ml_model, vectorizer = load_model()
except FileNotFoundError:
    train_and_save_model(df)
    ml_model, vectorizer = load_model()

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def assign_priority_label(score):
    if score >= 10:
        return "HIGH"
    elif score >= 6:
        return "MEDIUM"
    else:
        return "LOW"


def calculate_reason(email, dynamic_keywords):
    reasons = []
    sender = email["sender"].lower()
    subject = email["subject"].lower()
    body = email["body"].lower()

    if "boss" in sender or "bank" in sender or "placement" in sender:
        reasons.append("Sender=HighImportance")
    elif "hr" in sender:
        reasons.append("Sender=HR")
    elif "teammate" in sender:
        reasons.append("Sender=Teammate")

    for word in dynamic_keywords:
        if word in subject or word in body:
            reasons.append(f"Keyword='{word}'")

    if email["replied"] == 1:
        reasons.append("Replied")
    if email["opened"] == 1:
        reasons.append("Opened")
    if email["deleted"] == 1:
        reasons.append("Deleted")

    return ", ".join(reasons)

# --------------------------------------------------
# Rule-based Priority
# --------------------------------------------------
df["priority_rule_score"] = df.apply(
    lambda row: calculate_priority(row, dynamic_keywords=top_keywords),
    axis=1
)
df["priority_rule_label"] = df["priority_rule_score"].apply(assign_priority_label)

# --------------------------------------------------
# Load ML Model
# --------------------------------------------------

df["priority_ml_label"] = df.apply(
    lambda row: predict_priority_ml(row, ml_model, vectorizer),
    axis=1
)

df["reason_for_priority"] = df.apply(
    lambda row: calculate_reason(row, dynamic_keywords=top_keywords),
    axis=1
)

# --------------------------------------------------
# Load User Feedback
# --------------------------------------------------
UPDATE_FILE = "../output/updated_priorities.csv"

if os.path.exists(UPDATE_FILE):
    updates = pd.read_csv(UPDATE_FILE)
    df = df.merge(updates, on="email_id", how="left")
else:
    df["priority_updated"] = None

df["priority_final"] = df["priority_updated"].fillna(df["priority_rule_label"])
df_sorted = df.sort_values(by="priority_rule_score", ascending=False)

# --------------------------------------------------
# MAIN TABS (UNCHANGED STYLES & LOGIC)
# --------------------------------------------------
tab1, tab2 = st.tabs(["Overview", "Inbox & Filters"])

# =========================
# TAB 1 — Overview
# =========================
with tab1:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Emails", len(df))

    c2.markdown(f"<h3>🔴 {(df['priority_final']=='HIGH').sum()}</h3>", unsafe_allow_html=True)
    c3.markdown(f"<h3>🟠 {(df['priority_final']=='MEDIUM').sum()}</h3>", unsafe_allow_html=True)
    c4.markdown(f"<h3>🟢 {(df['priority_final']=='LOW').sum()}</h3>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(
        f"""
        <div style="background:#f0f2f6; padding:14px; border-radius:8px;">
            <h4>🔥 Top Dynamic Keywords</h4>
            <p style="font-size:16px;">{', '.join(top_keywords)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    left_info, right_info = st.columns(2)

    with left_info:
        st.markdown("""
        ### 📌 How This System Works
        - Rule-based priority engine
        - NLP keyword extraction
        - User feedback learning
        - Auto ML retraining
        """)

    with right_info:
        st.markdown("""
        ### 🛠 Technologies Used
        - Python, Pandas
        - Streamlit
        - Scikit-learn
        - NLP
        """)

# =========================
# TAB 2 — Inbox & Filters
# =========================
with tab2:
    left, right = st.columns([1, 3])

    with left:
        method = st.radio("Priority Method", ["Rule-Based", "ML-Based"])
        priority_filter = st.selectbox("Filter by Priority", ["All", "HIGH", "MEDIUM", "LOW"])
        sender_filter = st.selectbox("Filter by Sender", ["All"] + sorted(df["sender"].unique()))
        keyword_filter = st.selectbox("Filter by Keyword", ["All"] + top_keywords)
        top_n = st.slider("Top Emails", 5, len(df_sorted), 10)

    priority_col = "priority_rule_label" if method == "Rule-Based" else "priority_ml_label"

    df_view = df_sorted.copy()
    df_view["display_priority"] = df_view["priority_updated"].fillna(df_view[priority_col])

    if priority_filter != "All":
        df_view = df_view[df_view["display_priority"] == priority_filter]
    if sender_filter != "All":
        df_view = df_view[df_view["sender"] == sender_filter]
    if keyword_filter != "All":
        df_view = df_view[
            df_view["subject"].str.contains(keyword_filter, case=False) |
            df_view["body"].str.contains(keyword_filter, case=False)
        ]

    display_df = df_view.head(top_n)

    updated_priorities = {}

    with right:
        for _, row in display_df.iterrows():
            email_id = row["email_id"]
            priority = row["display_priority"]

            color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}.get(priority)

            st.markdown(
                f"""
                <div style="border:1px solid #ccc; padding:8px; margin-bottom:6px;">
                    <b>{row['sender']}</b> | {row['subject']} |
                    <span style="color:{color}; font-weight:600;">{priority}</span>
                    <div style="font-size:13px;color:#555;">{row['reason_for_priority']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            new_priority = st.selectbox(
                f"Update priority for Email ID {email_id}",
                ["No Change", "LOW", "MEDIUM", "HIGH"],
                key=f"priority_{email_id}"
            )

            if new_priority != "No Change":
                updated_priorities[email_id] = new_priority

    if st.button("💾 Save Updates & Retrain ML"):
        if updated_priorities:
            import os

            # Ensure output directory exists
            os.makedirs(os.path.dirname(UPDATE_FILE), exist_ok=True)

            updates_df = pd.DataFrame(
                updated_priorities.items(),
                columns=["email_id", "priority_updated"]
            )

            updates_df.to_csv(UPDATE_FILE, index=False)

            train_and_save_model(df)

            st.success("✅ Priorities saved & ML retrained")
        else:
            st.info("No changes made.")

