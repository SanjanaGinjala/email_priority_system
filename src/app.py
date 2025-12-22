import streamlit as st
import pandas as pd
import os

from load_data import load_emails
from priority_engine import calculate_priority
from nlp_keywords import extract_important_keywords
from ml_model import load_model, predict_priority_ml, train_and_save_model

# -------------------------
# Helper Functions
# -------------------------
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


# -------------------------
# Load Data
# -------------------------
df = load_emails()
top_keywords = extract_important_keywords(df, top_n=20)

# -------------------------
# Rule-based Priority
# -------------------------
df["priority_rule_score"] = df.apply(
    lambda row: calculate_priority(row, dynamic_keywords=top_keywords),
    axis=1
)
df["priority_rule_label"] = df["priority_rule_score"].apply(assign_priority_label)

# -------------------------
# Load ML Model
# -------------------------
ml_model, vectorizer = load_model()

df["priority_ml_label"] = df.apply(
    lambda row: predict_priority_ml(row, ml_model, vectorizer),
    axis=1
)

df["reason_for_priority"] = df.apply(
    lambda row: calculate_reason(row, dynamic_keywords=top_keywords),
    axis=1
)

# -------------------------
# Load User Updates (Feedback)
# -------------------------
UPDATE_FILE = "../output/updated_priorities.csv"
if os.path.exists(UPDATE_FILE):
    updates = pd.read_csv(UPDATE_FILE)
    df = df.merge(updates, on="email_id", how="left")
else:
    df["priority_updated"] = None

# Final priority to display
df["priority_final"] = df["priority_updated"].fillna(df["priority_rule_label"])

df_sorted = df.sort_values(by="priority_rule_score", ascending=False)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Email Priority Dashboard", layout="wide")
st.title("📧 Email Priority Dashboard")

tab1, tab2 = st.tabs(["Overview", "Inbox & Filters"])

# -------------------------
# TAB 1 — Overview (NO EMAILS HERE)
# -------------------------
# -------------------------
# TAB 1 — Overview (NO EMAILS HERE)
# -------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Emails", len(df))

    c2.markdown("""
    <div style="display:flex; align-items:center; gap:8px;">
        <div style="width:16px; height:16px; background:red; border-radius:50%;"></div>
        <b>HIGH Priority</b>
    </div>
    <h3>{}</h3>
    """.format((df["priority_final"] == "HIGH").sum()), unsafe_allow_html=True)

    c3.markdown("""
    <div style="display:flex; align-items:center; gap:8px;">
        <div style="width:16px; height:16px; background:orange; border-radius:50%;"></div>
        <b>MEDIUM Priority</b>
    </div>
    <h3>{}</h3>
    """.format((df["priority_final"] == "MEDIUM").sum()), unsafe_allow_html=True)

    c4.markdown("""
    <div style="display:flex; align-items:center; gap:8px;">
        <div style="width:16px; height:16px; background:green; border-radius:50%;"></div>
        <b>LOW Priority</b>
    </div>
    <h3>{}</h3>
    """.format((df["priority_final"] == "LOW").sum()), unsafe_allow_html=True)

    st.markdown("---")

    # 🔥 Dynamic Keywords (DO NOT TOUCH – as requested)
    st.markdown(
        f"""
        <div style="background:#f0f2f6; padding:14px; border-radius:8px;">
            <h4>🔥 Top Dynamic Keywords</h4>
            <p style="font-size:16px;">{', '.join(top_keywords)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")

    # 🔹 Additional informative sections (fills empty space cleanly)
    left_info, right_info = st.columns(2)

    with left_info:
        st.markdown("""
        ### 📌 How This System Works
        - Emails are analyzed using **rule-based logic**
        - Important keywords are extracted using **NLP**
        - User feedback improves accuracy over time
        - ML model retrains automatically
        """)

    with right_info:
        st.markdown("""
        ### 🛠 Technologies Used
        - Python & Pandas
        - Streamlit (Frontend)
        - Scikit-learn (ML)
        - NLP Keyword Extraction
        - CSV-based Feedback Learning
        """)

# -------------------------
# TAB 2 — Inbox & Filters
# -------------------------
with tab2:
    left, right = st.columns([1, 3])

    # ---------- Filters
    with left:
        st.markdown("### 🔍 Filters")

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

    # ---------- Email Cards
    updated_priorities = {}

    with right:
        for _, row in display_df.iterrows():
            email_id = row["email_id"]
            priority = row["display_priority"]

            # Priority text color
            priority_color = {
                "HIGH": "red",
                "MEDIUM": "orange",
                "LOW": "green"
            }.get(priority, "black")

            st.markdown(
                f"""
                <div style="
                    border: 1px solid #ccc;
                    padding: 6px 10px;
                    margin-bottom: 8px;
                    border-radius: 4px;
                    font-size: 14px;
                ">
                    <div>
                        <b>{row['sender']}</b> |
                        {row['subject']} |
                        <span style="color:{priority_color}; font-weight:600;">
                            {priority}
                        </span>
                    </div>
                    <div style="font-size:13px; color:#444; margin-top:3px;">
                        {row['reason_for_priority']}
                    </div>
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


    # ---------- Save & Retrain
    if st.button("💾 Save Updates & Retrain ML"):
        if updated_priorities:
            updates_df = pd.DataFrame(
                updated_priorities.items(),
                columns=["email_id", "priority_updated"]
            )

            if os.path.exists(UPDATE_FILE):
                old = pd.read_csv(UPDATE_FILE)
                updates_df = pd.concat([old, updates_df]).drop_duplicates(
                    subset="email_id", keep="last"
                )

            updates_df.to_csv(UPDATE_FILE, index=False)
            st.success("✅ Priorities saved")

            # Retrain ML using updated labels
            df["priority_updated"] = df["priority_updated"].fillna(df["priority_rule_label"])
            df["priority_rule_label"] = df["priority_updated"]

            train_and_save_model(df)
            st.success("🤖 ML retrained using user feedback")

        else:
            st.info("No changes made.")
