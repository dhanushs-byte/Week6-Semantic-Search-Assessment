"""Retrieval evaluation helpers: hit@k and mean reciprocal rank."""
def retrieval_metrics(rankings, expected):
    if len(rankings) != len(expected):
        raise ValueError("rankings and expected must have equal lengths")
    hits = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = []
    for results, target in zip(rankings, expected):
        sources = [r.get("source") for r in results]
        rank = next((i + 1 for i, source in enumerate(sources) if source == target), None)
        for k in hits:
            if rank is not None and rank <= k:
                hits[k] += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    n = max(1, len(expected))
    return {
        "hit@1": hits[1] / n, "hit@3": hits[3] / n,
        "hit@5": hits[5] / n,
        "mrr": sum(reciprocal_ranks) / n
    }
