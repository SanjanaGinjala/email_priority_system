from sklearn.feature_extraction.text import CountVectorizer

def extract_important_keywords(df, top_n=20):
    """
    Extracts top_n keywords from subject and body combined
    """
    text_data = df["subject"].fillna("") + " " + df["body"].fillna("")
    vectorizer = CountVectorizer(stop_words="english", max_features=500)
    X = vectorizer.fit_transform(text_data)
    sum_words = X.sum(axis=0)
    keywords_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    keywords_freq = sorted(keywords_freq, key=lambda x: x[1], reverse=True)
    top_keywords = [k[0] for k in keywords_freq[:top_n]]
    return top_keywords
