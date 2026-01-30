from collections import Counter

def document_coverage(evidence):
    """
    Calculates contribution percentage per document
    """
    sources = [e["source"] for e in evidence]
    total = len(sources)
    counts = Counter(sources)

    coverage = {}
    for doc, count in counts.items():
        coverage[doc] = round((count / total) * 100, 2)

    return coverage
