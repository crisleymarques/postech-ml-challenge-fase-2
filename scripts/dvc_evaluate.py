from __future__ import annotations

import json
import sys
from pathlib import Path


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

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
