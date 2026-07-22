from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.movielens_pipeline import GLOBAL_SEED
from src.evaluation.recommender_evaluation import GLOBAL_EVAL_K, run_benchmark
from src.models.baselines import ItemKNNRecommender, PopularityRecommender


def main() -> None:
    results = run_benchmark(
        recommenders=[
            PopularityRecommender(),
            ItemKNNRecommender(n_neighbors=40),
        ],
        processed_dir=PROJECT_ROOT / "data" / "processed" / "movielens",
        output_dir=PROJECT_ROOT / "data" / "processed" / "movielens" / "evaluation",
        k=GLOBAL_EVAL_K,
        seed=GLOBAL_SEED,
    )

    print(
        json.dumps(
            results["aggregate"].to_dict(orient="records"),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
