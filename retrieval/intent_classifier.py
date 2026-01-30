def classify_intent(query):
    q = query.lower()

    if any(w in q for w in ["compare", "difference", "vs", "contrast"]):
        return "comparison"

    if any(w in q for w in ["summarize", "summary", "overview"]):
        return "summarization"

    return "qa"
