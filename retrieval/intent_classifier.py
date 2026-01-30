def classify_intent(query):
    q = query.lower()
    if any(w in q for w in ["summarize", "summary", "overview"]):
        return "summarization"
    return "qa"
