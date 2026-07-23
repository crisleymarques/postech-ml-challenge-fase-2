from __future__ import annotations

import json
import pickle
from pathlib import Path

from src.models.baselines import BaseRecommender, ensure_output_dir


def _model_file_name(model_name: str) -> str:
    return f"{model_name}.pkl"


def save_trained_recommenders(
    recommenders: list[BaseRecommender],
    output_dir: str | Path,
    training_summary: dict,
) -> dict:
    output_dir = ensure_output_dir(output_dir)
    registry: list[dict[str, str]] = []

    for recommender in recommenders:
        file_name = _model_file_name(recommender.model_name)
        with (output_dir / file_name).open("wb") as model_file:
            pickle.dump(recommender, model_file)

        registry.append(
            {
                "model_name": recommender.model_name,
                "file_name": file_name,
                "class_name": recommender.__class__.__name__,
            }
        )

    payload = {
        "models": registry,
        "training_summary": training_summary,
    }
    with (output_dir / "model_registry.json").open("w", encoding="utf-8") as registry_file:
        json.dump(payload, registry_file, indent=2, ensure_ascii=False)

    return payload


def load_trained_recommenders(output_dir: str | Path) -> tuple[list[BaseRecommender], dict]:
    output_dir = Path(output_dir)
    registry_payload = json.loads((output_dir / "model_registry.json").read_text(encoding="utf-8"))

    recommenders: list[BaseRecommender] = []
    for model_metadata in registry_payload["models"]:
        with (output_dir / model_metadata["file_name"]).open("rb") as model_file:
            recommenders.append(pickle.load(model_file))

    return recommenders, registry_payload["training_summary"]
