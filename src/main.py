from load_data import load_emails
from priority_engine import calculate_priority
from nlp_keywords import extract_important_keywords
from ml_model import train_and_save_model

def main():
    df = load_emails()

    # Extract keywords
    top_keywords = extract_important_keywords(df, top_n=20)

    # Rule-based priority score
    df["priority_rule_score"] = df.apply(
        lambda row: calculate_priority(row, dynamic_keywords=top_keywords),
        axis=1
    )

    def assign_label(score):
        if score >= 10:
            return "HIGH"
        elif score >= 6:
            return "MEDIUM"
        else:
            return "LOW"

    df["priority_rule_label"] = df["priority_rule_score"].apply(assign_label)

    # Initial ML training
    train_and_save_model(df)

if __name__ == "__main__":
    main()
