from __future__ import annotations

import json
import sys
from pathlib import Path
import mlflow
import mlflow.sklearn
import mlflow.pytorch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.movielens_pipeline import set_global_seed
from src.evaluation.recommender_evaluation import run_benchmark_with_fitted
from src.models.model_persistence import load_trained_recommenders
from src.pipeline_config import load_params, resolve_path


def main() -> None:
    params = load_params(PROJECT_ROOT / "params.yaml")
    seed = int(params["seed"])
    processed_dir = resolve_path(PROJECT_ROOT, params["data"]["processed_dir"])
    model_dir = resolve_path(PROJECT_ROOT, params["data"]["model_dir"])
    evaluation_dir = resolve_path(PROJECT_ROOT, params["data"]["evaluation_dir"])
    metrics_path = resolve_path(PROJECT_ROOT, params["reports"]["evaluation_metrics"])

    set_global_seed(seed)
    recommenders, training_summary = load_trained_recommenders(model_dir)
    results = run_benchmark_with_fitted(
        recommenders=recommenders,
        processed_dir=processed_dir,
        output_dir=evaluation_dir,
        k=int(params["evaluate"]["k"]),
        seed=seed,
    )

    aggregate_records = results["aggregate"].to_dict(orient="records")
    metrics = {
        "seed": seed,
        "k": int(params["evaluate"]["k"]),
        "training_summary": training_summary,
        "results": {row["model_name"]: row for row in aggregate_records},
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("MovieLens_Recommender_Experiment")
    for model_name, model_results in metrics["results"].items():
        with mlflow.start_run(run_name=f"Avaliacao_{model_name}"):
            mlflow.log_params({
                "seed": params["seed"],
                "evaluate_k": int(params["evaluate"]["k"]),
                "model_name": model_name
            })
            mlflow.log_metrics({
                "ndcg_at_k": model_results.get("ndcg", 0.0),
                "precision_at_k": model_results.get("precision", 0.0),
                "recall_at_k": model_results.get("recall", 0.0)
            })
            mlflow.set_tag("pipeline_stage", "dvc_evaluate")
            mlflow.log_artifact(str(metrics_path))
            class_map = {
                "popularity": "PopularityRecommender",
                "item_knn_sklearn": "ItemKNNRecommender"
            }
            target_class = class_map.get(model_name, model_name)

            model_obj = next((m for m in recommenders if m.__class__.__name__ == target_class), None)

            if model_obj:
                if "NCFRecommender" in str(type(model_obj)) or "Neural" in model_name:
                    mlflow.pytorch.log_model(model_obj, artifact_path=f"modelo_{model_name}")
                else:
                    mlflow.sklearn.log_model(
                        sk_model=model_obj,
                        artifact_path=f"modelo_{model_name}",
                        serialization_format="cloudpickle"
                    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
