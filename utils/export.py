def generate_report(query, answer, evidence, confidence):
    report = []
    report.append("=== MULTIMODAL RAG REPORT ===\n")
    report.append(f"Query:\n{query}\n")
    report.append(f"Answer:\n{answer}\n")
    report.append(f"Confidence: {int(confidence*100)}%\n\n")

    report.append("Evidence Used:\n")
    for i, e in enumerate(evidence, 1):
        report.append(
            f"{i}. {e['source']} | modality={e['modality']}\n"
            f"   Content: {e['content'][:300]}\n"
        )

    return "\n".join(report)
