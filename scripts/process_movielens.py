from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.movielens_pipeline import GLOBAL_SEED, run_pipeline


def main() -> None:
    results = run_pipeline(
        raw_dir=PROJECT_ROOT / "data" / "raw",
        output_dir=PROJECT_ROOT / "data" / "processed" / "movielens",
        seed=GLOBAL_SEED,
    )

    print(
        json.dumps(
            {
                "artifact_summary": results["artifact_summary"],
                "raw_summary": results["raw_profile"]["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
