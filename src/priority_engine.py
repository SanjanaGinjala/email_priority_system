def calculate_priority(email_row, dynamic_keywords=[]):
    """
    Returns a numeric priority score for an email
    """
    score = 0
    sender = email_row["sender"].lower()
    subject = email_row["subject"].lower()
    body = email_row["body"].lower()

    # Sender importance
    if "boss" in sender or "bank" in sender or "placement" in sender:
        score += 5
    elif "hr" in sender:
        score += 3
    elif "teammate" in sender:
        score += 2

    # Keyword importance
    for word in dynamic_keywords:
        if word in subject or word in body:
            score += 2

    # Email actions
    if email_row["replied"] == 1:
        score += 1
    if email_row["opened"] == 1:
        score += 1
    if email_row["deleted"] == 1:
        score -= 1

    return score
