from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.movielens_pipeline import (
    build_feature_tables,
    load_clean_artifacts,
    persist_processed_artifacts,
    set_global_seed,
    temporal_leave_last_k_out,
)
from src.pipeline_config import load_params, resolve_path


def main() -> None:
    params = load_params(PROJECT_ROOT / "params.yaml")
    seed = int(params["seed"])
    interim_dir = resolve_path(PROJECT_ROOT, params["data"]["interim_dir"])
    processed_dir = resolve_path(PROJECT_ROOT, params["data"]["processed_dir"])
    metrics_path = resolve_path(PROJECT_ROOT, params["reports"]["feature_eng_metrics"])

    set_global_seed(seed)
    artifacts = load_clean_artifacts(interim_dir)
    feature_tables = build_feature_tables(artifacts["bundle"])
    splits = temporal_leave_last_k_out(
        feature_tables["interactions"],
        validation_k=int(params["split"]["validation_k"]),
        test_k=int(params["split"]["test_k"]),
    )
    artifact_summary = persist_processed_artifacts(
        feature_tables=feature_tables,
        splits=splits,
        output_dir=processed_dir,
        quality_report={
            "raw_profile": artifacts["raw_profile"],
            "cleaning": artifacts["quality_report"],
        },
        seed=seed,
    )

    metrics = {
        "seed": seed,
        "n_interactions_full": artifact_summary["n_interactions_full"],
        "n_train": artifact_summary["n_train"],
        "n_validation": artifact_summary["n_validation"],
        "n_test": artifact_summary["n_test"],
        "n_users": artifact_summary["n_users"],
        "n_items": artifact_summary["n_items"],
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)

    print(json.dumps({"artifact_summary": artifact_summary, "metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
