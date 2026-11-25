"""Binary relevance construction and ranking metrics for held-out songs."""

import math
from numbers import Integral


def build_test_relevance(edges, test_ids):
    """Mirror observed edges and retain only positives internal to test IDs."""
    test_ids = set(test_ids)
    relevance = {}
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        if source == target or source not in test_ids or target not in test_ids:
            continue
        relevance.setdefault(source, set()).add(target)
        relevance.setdefault(target, set()).add(source)
    return relevance


def evaluate_rankings(rankings, relevance, k=10):
    """Return macro Recall@K, HitRate@K, MRR@K, and binary NDCG@K.

    Only queries with at least one observed positive are eligible. Recall uses
    all observed positives as its denominator; HitRate is one if any positive
    occurs in the top K; reciprocal rank uses the first hit; NDCG uses binary
    relevance and DCG discount 1/log2(rank + 1), with 1-based ranks.
    """
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 0:
        raise ValueError("k must be a non-negative integer")
    k = int(k)
    eligible = [(query_id, positives) for query_id, positives in relevance.items() if positives]
    query_count = len(eligible)
    if query_count == 0:
        recall = hit_rate = mrr = ndcg = 0.0
    else:
        recalls = []
        hits = []
        reciprocal_ranks = []
        ndcgs = []
        for query_id, positives in eligible:
            ranked = rankings.get(query_id, [])[:k]
            relevant_ranks = [
                rank for rank, item in enumerate(ranked, start=1)
                if item["song_id"] in positives
            ]
            recalls.append(len(relevant_ranks) / len(positives))
            hits.append(float(bool(relevant_ranks)))
            reciprocal_ranks.append(1.0 / relevant_ranks[0] if relevant_ranks else 0.0)
            dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
            ideal_count = min(len(positives), k)
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
            ndcgs.append(dcg / idcg if idcg else 0.0)
        recall = sum(recalls) / query_count
        hit_rate = sum(hits) / query_count
        mrr = sum(reciprocal_ranks) / query_count
        ndcg = sum(ndcgs) / query_count

    return {
        f"Recall@{k}": recall,
        f"HitRate@{k}": hit_rate,
        f"MRR@{k}": mrr,
        f"NDCG@{k}": ndcg,
        "query_count": query_count,
        "eligible_positive_edges": len(
            {
                frozenset((query_id, target_id))
                for query_id, positives in eligible
                for target_id in positives
            }
        ),
    }
