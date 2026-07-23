from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.movielens_pipeline import (
    clean_movielens_data,
    load_movielens_data,
    persist_clean_artifacts,
    profile_raw_data,
    set_global_seed,
)
from src.pipeline_config import load_params, resolve_path


def main() -> None:
    params = load_params(PROJECT_ROOT / "params.yaml")
    seed = int(params["seed"])
    raw_dir = resolve_path(PROJECT_ROOT, params["data"]["raw_dir"])
    interim_dir = resolve_path(PROJECT_ROOT, params["data"]["interim_dir"])
    metrics_path = resolve_path(PROJECT_ROOT, params["reports"]["preprocess_metrics"])

    set_global_seed(seed)
    bundle = load_movielens_data(raw_dir)
    raw_profile = profile_raw_data(bundle)
    clean_bundle, quality_report = clean_movielens_data(bundle)
    preprocess_summary = persist_clean_artifacts(
        bundle=clean_bundle,
        raw_profile=raw_profile,
        quality_report=quality_report,
        output_dir=interim_dir,
        seed=seed,
    )

    metrics = {
        "seed": seed,
        "n_ratings": preprocess_summary["n_ratings"],
        "n_users": preprocess_summary["n_users"],
        "n_movies": preprocess_summary["n_movies"],
        "ratings_removed": quality_report["rows_removed"]["ratings"],
        "users_removed": quality_report["rows_removed"]["users"],
        "movies_removed": quality_report["rows_removed"]["movies"],
        "sparsity": raw_profile["summary"]["sparsity"],
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)

    print(json.dumps({"preprocess_summary": preprocess_summary, "metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
