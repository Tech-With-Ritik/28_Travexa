def confidence_score(results, intent="qa"):
    """
    Returns confidence between 0 and 1
    Intent-aware confidence scoring
    """

    if not results:
        return 0.0

    # Summarization confidence depends on document availability
    if intent == "summarization":
        return 0.85

    # QA confidence depends on number of supporting chunks
    n = len(results)

    if n >= 5:
        return 0.9
    elif n >= 3:
        return 0.75
    elif n == 2:
        return 0.55
    else:
        return 0.35


def uncertainty_message(confidence: float):
    """
    Returns a human-readable uncertainty message
    """

    if confidence >= 0.8:
        return None
    elif confidence >= 0.5:
        return "⚠️ This answer is based on limited evidence."
    else:
        return "⚠️ High uncertainty: evidence is weak or incomplete."
