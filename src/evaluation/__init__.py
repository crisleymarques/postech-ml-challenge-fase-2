"""Rotinas comuns de avaliação para recomendação."""

from .recommender_evaluation import (
    GLOBAL_EVAL_K,
    catalog_coverage_at_k,
    evaluate_recommender,
    hit_rate_at_k,
    load_processed_movielens_artifacts,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    run_benchmark,
)

__all__ = [
    "GLOBAL_EVAL_K",
    "catalog_coverage_at_k",
    "evaluate_recommender",
    "hit_rate_at_k",
    "load_processed_movielens_artifacts",
    "mrr_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "run_benchmark",
]
