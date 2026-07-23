from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.movielens_pipeline import set_global_seed
from src.evaluation.recommender_evaluation import load_processed_movielens_artifacts
from src.models.baselines import ItemKNNRecommender, PopularityRecommender
from src.models.model_persistence import save_trained_recommenders
from src.pipeline_config import load_params, resolve_path


def main() -> None:
    params = load_params(PROJECT_ROOT / "params.yaml")
    seed = int(params["seed"])
    processed_dir = resolve_path(PROJECT_ROOT, params["data"]["processed_dir"])
    model_dir = resolve_path(PROJECT_ROOT, params["data"]["model_dir"])
    metrics_path = resolve_path(PROJECT_ROOT, params["reports"]["train_metrics"])

    set_global_seed(seed)
    artifacts = load_processed_movielens_artifacts(processed_dir)

    if params["train"]["use_validation_in_fit"]:
        fit_interactions = pd.concat([artifacts["train"], artifacts["validation"]], ignore_index=True)
        fit_split = "train+validation"
    else:
        fit_interactions = artifacts["train"]
        fit_split = "train"

    recommenders = []
    if params["train"]["baselines"]["popularity"]:
        recommenders.append(PopularityRecommender())
    if params["train"]["baselines"]["item_knn_sklearn"]:
        recommenders.append(
            ItemKNNRecommender(n_neighbors=int(params["train"]["item_knn"]["n_neighbors"]))
        )

    for recommender in recommenders:
        recommender.fit(fit_interactions, artifacts["user_features"], artifacts["item_features"])

    training_summary = {
        "seed": seed,
        "fit_split": fit_split,
        "n_fit_interactions": int(len(fit_interactions)),
        "n_users": int(artifacts["metadata"]["n_users"]),
        "n_items": int(artifacts["metadata"]["n_items"]),
        "models": [recommender.model_name for recommender in recommenders],
    }
    if params["train"]["baselines"]["item_knn_sklearn"]:
        training_summary["item_knn_n_neighbors"] = int(params["train"]["item_knn"]["n_neighbors"])

    registry = save_trained_recommenders(
        recommenders=recommenders,
        output_dir=model_dir,
        training_summary=training_summary,
    )

    metrics = {
        "seed": seed,
        "fit_split": fit_split,
        "n_fit_interactions": int(len(fit_interactions)),
        "n_models_trained": len(recommenders),
        "item_knn_n_neighbors": int(params["train"]["item_knn"]["n_neighbors"]),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)

    print(json.dumps({"training_summary": training_summary, "registry": registry, "metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
