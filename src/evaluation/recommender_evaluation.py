from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.movielens_pipeline import GLOBAL_SEED, set_global_seed
from src.models.baselines import BaseRecommender, ensure_output_dir


GLOBAL_EVAL_K = 10


def load_processed_movielens_artifacts(processed_dir: str | Path) -> dict[str, pd.DataFrame | dict]:
    processed_dir = Path(processed_dir)
    metadata = json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))

    return {
        "train": pd.read_csv(processed_dir / "train_interactions.csv"),
        "validation": pd.read_csv(processed_dir / "validation_interactions.csv"),
        "test": pd.read_csv(processed_dir / "test_interactions.csv"),
        "user_features": pd.read_csv(processed_dir / "user_features.csv"),
        "item_features": pd.read_csv(processed_dir / "item_features.csv"),
        "metadata": metadata,
    }


def precision_at_k(recommended_items: list[int], relevant_items: set[int], k: int) -> float:
    if k <= 0:
        raise ValueError("k precisa ser positivo.")
    if not recommended_items:
        return 0.0
    hits = sum(1 for item in recommended_items[:k] if item in relevant_items)
    return hits / k


def recall_at_k(recommended_items: list[int], relevant_items: set[int], k: int) -> float:
    if k <= 0:
        raise ValueError("k precisa ser positivo.")
    if not relevant_items:
        return 0.0
    hits = sum(1 for item in recommended_items[:k] if item in relevant_items)
    return hits / len(relevant_items)


def hit_rate_at_k(recommended_items: list[int], relevant_items: set[int], k: int) -> float:
    return float(any(item in relevant_items for item in recommended_items[:k]))


def ndcg_at_k(recommended_items: list[int], relevant_items: set[int], k: int) -> float:
    dcg = 0.0
    for rank, item in enumerate(recommended_items[:k], start=1):
        if item in relevant_items:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(relevant_items), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def mrr_at_k(recommended_items: list[int], relevant_items: set[int], k: int) -> float:
    for rank, item in enumerate(recommended_items[:k], start=1):
        if item in relevant_items:
            return 1.0 / rank
    return 0.0


def catalog_coverage_at_k(recommendations_by_user: dict[int, list[int]], all_item_ids: list[int], k: int) -> float:
    if not all_item_ids:
        return 0.0
    recommended_items = {item for items in recommendations_by_user.values() for item in items[:k]}
    return len(recommended_items) / len(set(all_item_ids))


def _build_seen_items(interactions: pd.DataFrame) -> dict[int, set[int]]:
    return (
        interactions.groupby("user_id")["movie_id"]
        .agg(lambda series: set(series.astype(int).tolist()))
        .to_dict()
    )


def _build_ground_truth(interactions: pd.DataFrame) -> dict[int, set[int]]:
    return (
        interactions.groupby("user_id")["movie_id"]
        .agg(lambda series: set(series.astype(int).tolist()))
        .to_dict()
    )


def evaluate_fitted_recommender(
    recommender: BaseRecommender,
    fit_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    item_features: pd.DataFrame,
    k: int = GLOBAL_EVAL_K,
) -> tuple[dict, pd.DataFrame]:
    if k <= 0:
        raise ValueError("k precisa ser positivo.")

    all_item_ids = item_features["movie_id"].astype(int).tolist()
    seen_items = _build_seen_items(fit_interactions)
    ground_truth = _build_ground_truth(test_interactions)

    per_user_rows: list[dict] = []
    recommendations_by_user: dict[int, list[int]] = {}

    for user_id in sorted(ground_truth):
        user_seen = seen_items.get(user_id, set())
        candidate_items = [item_id for item_id in all_item_ids if item_id not in user_seen]
        recommendations = recommender.recommend(user_id=user_id, candidate_item_ids=candidate_items, k=k)
        relevant_items = ground_truth[user_id]
        recommendations_by_user[user_id] = recommendations

        per_user_rows.append(
            {
                "model_name": recommender.model_name,
                "user_id": user_id,
                "precision_at_k": precision_at_k(recommendations, relevant_items, k),
                "recall_at_k": recall_at_k(recommendations, relevant_items, k),
                "hit_rate_at_k": hit_rate_at_k(recommendations, relevant_items, k),
                "ndcg_at_k": ndcg_at_k(recommendations, relevant_items, k),
                "mrr_at_k": mrr_at_k(recommendations, relevant_items, k),
                "recommended_items": recommendations,
                "relevant_items": sorted(relevant_items),
            }
        )

    per_user_results = pd.DataFrame(per_user_rows)
    aggregate_results = {
        "model_name": recommender.model_name,
        "k": k,
        "n_users_evaluated": int(len(per_user_results)),
        "precision_at_k": float(per_user_results["precision_at_k"].mean()),
        "recall_at_k": float(per_user_results["recall_at_k"].mean()),
        "hit_rate_at_k": float(per_user_results["hit_rate_at_k"].mean()),
        "ndcg_at_k": float(per_user_results["ndcg_at_k"].mean()),
        "mrr_at_k": float(per_user_results["mrr_at_k"].mean()),
        "catalog_coverage_at_k": float(catalog_coverage_at_k(recommendations_by_user, all_item_ids, k)),
    }

    return aggregate_results, per_user_results


def evaluate_recommender(
    recommender: BaseRecommender,
    fit_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    k: int = GLOBAL_EVAL_K,
) -> tuple[dict, pd.DataFrame]:
    if k <= 0:
        raise ValueError("k precisa ser positivo.")

    recommender.fit(fit_interactions, user_features, item_features)
    return evaluate_fitted_recommender(
        recommender=recommender,
        fit_interactions=fit_interactions,
        test_interactions=test_interactions,
        item_features=item_features,
        k=k,
    )

def run_benchmark(
    recommenders: list[BaseRecommender],
    processed_dir: str | Path = "data/processed/movielens",
    output_dir: str | Path = "data/processed/movielens/evaluation",
    k: int = GLOBAL_EVAL_K,
    seed: int = GLOBAL_SEED,
) -> dict[str, pd.DataFrame]:
    set_global_seed(seed)

    dataset = load_processed_movielens_artifacts(processed_dir)
    fit_interactions = pd.concat([dataset["train"], dataset["validation"]], ignore_index=True)
    test_interactions = dataset["test"]
    user_features = dataset["user_features"]
    item_features = dataset["item_features"]

    evaluation_output_dir = ensure_output_dir(output_dir)
    aggregate_rows: list[dict] = []
    detailed_frames: list[pd.DataFrame] = []

    for recommender in recommenders:
        aggregate_result, per_user_results = evaluate_recommender(
            recommender=recommender,
            fit_interactions=fit_interactions,
            test_interactions=test_interactions,
            user_features=user_features,
            item_features=item_features,
            k=k,
        )
        aggregate_rows.append(aggregate_result)
        detailed_frames.append(per_user_results)

    aggregate_df = pd.DataFrame(aggregate_rows).sort_values(
        by=["ndcg_at_k", "mrr_at_k", "catalog_coverage_at_k"],
        ascending=[False, False, False],
    )
    detailed_df = pd.concat(detailed_frames, ignore_index=True)

    aggregate_df.to_csv(evaluation_output_dir / "baseline_results.csv", index=False)
    detailed_df.to_json(
        evaluation_output_dir / "baseline_recommendations.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )

    protocol = {
        "seed": seed,
        "k": k,
        "fit_split": "train+validation",
        "evaluation_split": "test",
        "same_conditions": True,
        "metrics": [
            "precision_at_k",
            "recall_at_k",
            "hit_rate_at_k",
            "ndcg_at_k",
            "mrr_at_k",
            "catalog_coverage_at_k",
        ],
    }
    with (evaluation_output_dir / "evaluation_protocol.json").open("w", encoding="utf-8") as protocol_file:
        json.dump(protocol, protocol_file, indent=2, ensure_ascii=False)

    return {
        "aggregate": aggregate_df,
        "detailed": detailed_df,
    }


def run_benchmark_with_fitted(
    recommenders: list[BaseRecommender],
    processed_dir: str | Path = "data/processed/movielens",
    output_dir: str | Path = "data/processed/movielens/evaluation",
    k: int = GLOBAL_EVAL_K,
    seed: int = GLOBAL_SEED,
) -> dict[str, pd.DataFrame]:
    set_global_seed(seed)

    dataset = load_processed_movielens_artifacts(processed_dir)
    fit_interactions = pd.concat([dataset["train"], dataset["validation"]], ignore_index=True)
    test_interactions = dataset["test"]
    item_features = dataset["item_features"]

    evaluation_output_dir = ensure_output_dir(output_dir)
    aggregate_rows: list[dict] = []
    detailed_frames: list[pd.DataFrame] = []

    for recommender in recommenders:
        aggregate_result, per_user_results = evaluate_fitted_recommender(
            recommender=recommender,
            fit_interactions=fit_interactions,
            test_interactions=test_interactions,
            item_features=item_features,
            k=k,
        )
        aggregate_rows.append(aggregate_result)
        detailed_frames.append(per_user_results)

    aggregate_df = pd.DataFrame(aggregate_rows).sort_values(
        by=["ndcg_at_k", "mrr_at_k", "catalog_coverage_at_k"],
        ascending=[False, False, False],
    )
    detailed_df = pd.concat(detailed_frames, ignore_index=True)

    aggregate_df.to_csv(evaluation_output_dir / "baseline_results.csv", index=False)
    detailed_df.to_json(
        evaluation_output_dir / "baseline_recommendations.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )

    protocol = {
        "seed": seed,
        "k": k,
        "fit_split": "train+validation",
        "evaluation_split": "test",
        "same_conditions": True,
        "models_are_pretrained": True,
        "metrics": [
            "precision_at_k",
            "recall_at_k",
            "hit_rate_at_k",
            "ndcg_at_k",
            "mrr_at_k",
            "catalog_coverage_at_k",
        ],
    }
    with (evaluation_output_dir / "evaluation_protocol.json").open("w", encoding="utf-8") as protocol_file:
        json.dump(protocol, protocol_file, indent=2, ensure_ascii=False)

    return {
        "aggregate": aggregate_df,
        "detailed": detailed_df,
    }
