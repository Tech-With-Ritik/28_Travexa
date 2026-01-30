def detect_conflicts(evidence):
    """
    Detect conflicting statements across retrieved chunks.
    Simple but effective heuristic for hackathon.
    """

    if len(evidence) < 2:
        return False, []

    texts = [e["content"].lower() for e in evidence]

    conflicts = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            # very simple contradiction heuristic
            if (
                "not" in texts[i] and "is" in texts[j]
                or "false" in texts[i] and "true" in texts[j]
            ):
                conflicts.append((evidence[i], evidence[j]))

    return len(conflicts) > 0, conflicts
