from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, OneHotEncoder

    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    SKLEARN_AVAILABLE = False

    class MinMaxScaler:  # type: ignore[no-redef]
        def fit_transform(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
            values = np.asarray(data, dtype=float)
            col_min = np.nanmin(values, axis=0)
            col_max = np.nanmax(values, axis=0)
            scale = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
            return (values - col_min) / scale

    class OneHotEncoder:  # type: ignore[no-redef]
        def __init__(self, handle_unknown: str = "ignore", sparse_output: bool = False, sparse: bool = False):
            self.handle_unknown = handle_unknown
            self.sparse_output = sparse_output
            self.sparse = sparse
            self.categories_: list[np.ndarray] = []

        def fit_transform(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
            frame = pd.DataFrame(data).copy()
            encoded_parts: list[np.ndarray] = []
            self.categories_ = []

            for column in frame.columns:
                categories = np.array(sorted(frame[column].astype(str).unique().tolist()), dtype=object)
                self.categories_.append(categories)
                mapping = {category: idx for idx, category in enumerate(categories)}
                encoded = np.zeros((len(frame), len(categories)), dtype=float)

                for row_idx, value in enumerate(frame[column].astype(str)):
                    category_idx = mapping.get(value)
                    if category_idx is not None:
                        encoded[row_idx, category_idx] = 1.0

                encoded_parts.append(encoded)

            if not encoded_parts:
                return np.empty((len(frame), 0), dtype=float)

            return np.concatenate(encoded_parts, axis=1)

    class MultiLabelBinarizer:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.classes_: np.ndarray = np.array([], dtype=object)

        def fit_transform(self, data: pd.Series | list[list[str]]) -> np.ndarray:
            labels = [list(values) for values in data]
            classes = sorted({label for values in labels for label in values})
            self.classes_ = np.array(classes, dtype=object)
            mapping = {label: idx for idx, label in enumerate(classes)}
            encoded = np.zeros((len(labels), len(classes)), dtype=int)

            for row_idx, values in enumerate(labels):
                for value in values:
                    encoded[row_idx, mapping[value]] = 1

            return encoded


GLOBAL_SEED = 42
RATINGS_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
MOVIES_COLUMNS = ["movie_id", "title", "genres"]
USERS_COLUMNS = ["user_id", "gender", "age", "occupation", "zip_code"]
TITLE_YEAR_REGEX = re.compile(r"\((\d{4})\)\s*$")


@dataclass(slots=True)
class MovieLensBundle:
    ratings: pd.DataFrame
    users: pd.DataFrame
    movies: pd.DataFrame


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _read_dat_file(file_path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(
        file_path,
        sep="::",
        engine="python",
        names=columns,
        encoding="latin-1",
    )


def _make_one_hot_encoder() -> Any:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _extract_release_year(title: str) -> float:
    if not isinstance(title, str):
        return np.nan

    match = TITLE_YEAR_REGEX.search(title)
    return float(match.group(1)) if match else np.nan


def _slugify_genre_name(genre: str) -> str:
    return genre.lower().replace("-", "_").replace(" ", "_").replace("'", "")


def load_movielens_data(raw_dir: str | Path) -> MovieLensBundle:
    raw_dir = Path(raw_dir)

    ratings = _read_dat_file(raw_dir / "ratings.dat", RATINGS_COLUMNS)
    users = _read_dat_file(raw_dir / "users.dat", USERS_COLUMNS)
    movies = _read_dat_file(raw_dir / "movies.dat", MOVIES_COLUMNS)

    return MovieLensBundle(ratings=ratings, users=users, movies=movies)


def profile_raw_data(bundle: MovieLensBundle) -> dict:
    ratings = bundle.ratings.copy()
    users = bundle.users.copy()
    movies = bundle.movies.copy()

    missing = {
        "ratings": ratings.isna().sum().to_dict(),
        "users": users.isna().sum().to_dict(),
        "movies": movies.isna().sum().to_dict(),
    }

    duplicate_rows = {
        "ratings_full_rows": int(ratings.duplicated().sum()),
        "ratings_user_movie_pairs": int(ratings.duplicated(subset=["user_id", "movie_id"]).sum()),
        "users_user_id": int(users.duplicated(subset=["user_id"]).sum()),
        "movies_movie_id": int(movies.duplicated(subset=["movie_id"]).sum()),
    }

    invalid_references = {
        "ratings_without_user": int((~ratings["user_id"].isin(users["user_id"])).sum()),
        "ratings_without_movie": int((~ratings["movie_id"].isin(movies["movie_id"])).sum()),
    }

    summary = {
        "n_ratings": int(len(ratings)),
        "n_users": int(ratings["user_id"].nunique()),
        "n_items": int(ratings["movie_id"].nunique()),
        "ratings_scale": sorted(ratings["rating"].dropna().unique().tolist()),
        "timestamp_min": int(ratings["timestamp"].min()),
        "timestamp_max": int(ratings["timestamp"].max()),
        "sparsity": float(
            1.0 - (len(ratings) / (ratings["user_id"].nunique() * ratings["movie_id"].nunique()))
        ),
    }

    return {
        "summary": summary,
        "missing_values": missing,
        "duplicate_rows": duplicate_rows,
        "invalid_references": invalid_references,
    }


def clean_movielens_data(bundle: MovieLensBundle) -> tuple[MovieLensBundle, dict]:
    ratings = bundle.ratings.copy()
    users = bundle.users.copy()
    movies = bundle.movies.copy()

    quality_report = {
        "initial_rows": {
            "ratings": int(len(ratings)),
            "users": int(len(users)),
            "movies": int(len(movies)),
        }
    }

    ratings = ratings.dropna(subset=RATINGS_COLUMNS)
    users = users.dropna(subset=USERS_COLUMNS)
    movies = movies.dropna(subset=MOVIES_COLUMNS)

    ratings = ratings.astype({"user_id": int, "movie_id": int, "rating": float, "timestamp": int})
    users = users.astype({"user_id": int, "gender": str, "age": int, "occupation": int, "zip_code": str})
    movies = movies.astype({"movie_id": int, "title": str, "genres": str})

    ratings = ratings[ratings["rating"].between(1, 5)]
    ratings = ratings.sort_values(["user_id", "movie_id", "timestamp"])
    ratings = ratings.drop_duplicates(subset=["user_id", "movie_id"], keep="last")

    users = users.drop_duplicates(subset=["user_id"], keep="last")
    movies = movies.drop_duplicates(subset=["movie_id"], keep="last")

    users["zip_code"] = users["zip_code"].str.strip()
    users["zip_prefix"] = users["zip_code"].str.extract(r"^(\d{3})", expand=False).fillna("unknown")

    movies["title"] = movies["title"].str.strip()
    movies["release_year"] = movies["title"].map(_extract_release_year)
    movies["release_year"] = movies["release_year"].fillna(movies["release_year"].median())
    movies["genres_list"] = movies["genres"].str.split("|")

    ratings["timestamp"] = ratings["timestamp"].astype(int)
    ratings["rated_at"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    ratings["label"] = 1

    valid_user_ids = set(users["user_id"])
    valid_movie_ids = set(movies["movie_id"])
    ratings = ratings[
        ratings["user_id"].isin(valid_user_ids) & ratings["movie_id"].isin(valid_movie_ids)
    ].copy()

    quality_report["final_rows"] = {
        "ratings": int(len(ratings)),
        "users": int(len(users)),
        "movies": int(len(movies)),
    }
    quality_report["rows_removed"] = {
        "ratings": quality_report["initial_rows"]["ratings"] - quality_report["final_rows"]["ratings"],
        "users": quality_report["initial_rows"]["users"] - quality_report["final_rows"]["users"],
        "movies": quality_report["initial_rows"]["movies"] - quality_report["final_rows"]["movies"],
    }
    quality_report["remaining_missing_values"] = {
        "ratings": ratings.isna().sum().to_dict(),
        "users": users.isna().sum().to_dict(),
        "movies": movies.isna().sum().to_dict(),
    }

    return MovieLensBundle(ratings=ratings, users=users, movies=movies), quality_report


def persist_clean_artifacts(
    bundle: MovieLensBundle,
    raw_profile: dict,
    quality_report: dict,
    output_dir: str | Path,
    seed: int = GLOBAL_SEED,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle.ratings.to_csv(output_dir / "ratings_clean.csv", index=False)
    bundle.users.to_csv(output_dir / "users_clean.csv", index=False)
    bundle.movies.to_csv(output_dir / "movies_clean.csv", index=False)

    preprocess_summary = {
        "seed": seed,
        "n_ratings": int(len(bundle.ratings)),
        "n_users": int(len(bundle.users)),
        "n_movies": int(len(bundle.movies)),
        "raw_profile": raw_profile,
        "quality_report": quality_report,
    }

    with (output_dir / "raw_profile.json").open("w", encoding="utf-8") as raw_profile_file:
        json.dump(raw_profile, raw_profile_file, indent=2, ensure_ascii=False)

    with (output_dir / "quality_report.json").open("w", encoding="utf-8") as quality_report_file:
        json.dump(quality_report, quality_report_file, indent=2, ensure_ascii=False)

    with (output_dir / "preprocess_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(preprocess_summary, summary_file, indent=2, ensure_ascii=False)

    return preprocess_summary


def load_clean_artifacts(input_dir: str | Path) -> dict[str, MovieLensBundle | dict]:
    input_dir = Path(input_dir)
    ratings = pd.read_csv(input_dir / "ratings_clean.csv", parse_dates=["rated_at"])
    users = pd.read_csv(input_dir / "users_clean.csv")
    movies = pd.read_csv(input_dir / "movies_clean.csv")

    if "genres_list" not in movies.columns:
        movies["genres_list"] = movies["genres"].astype(str).str.split("|")
    else:
        movies["genres_list"] = movies["genres"].astype(str).str.split("|")

    raw_profile = json.loads((input_dir / "raw_profile.json").read_text(encoding="utf-8"))
    quality_report = json.loads((input_dir / "quality_report.json").read_text(encoding="utf-8"))
    preprocess_summary = json.loads((input_dir / "preprocess_summary.json").read_text(encoding="utf-8"))

    return {
        "bundle": MovieLensBundle(ratings=ratings, users=users, movies=movies),
        "raw_profile": raw_profile,
        "quality_report": quality_report,
        "preprocess_summary": preprocess_summary,
    }


def build_feature_tables(bundle: MovieLensBundle) -> dict[str, pd.DataFrame]:
    ratings = bundle.ratings.copy()
    users = bundle.users.copy()
    movies = bundle.movies.copy()

    unique_user_ids = sorted(ratings["user_id"].unique().tolist())
    unique_movie_ids = sorted(ratings["movie_id"].unique().tolist())
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}

    ratings["user_idx"] = ratings["user_id"].map(user_to_idx)
    ratings["item_idx"] = ratings["movie_id"].map(movie_to_idx)

    user_stats = (
        ratings.groupby("user_id", as_index=False)
        .agg(interaction_count=("movie_id", "size"), mean_rating=("rating", "mean"))
    )
    item_stats = (
        ratings.groupby("movie_id", as_index=False)
        .agg(interaction_count=("user_id", "size"), mean_rating=("rating", "mean"))
    )

    users = users.merge(user_stats, on="user_id", how="inner")
    users["user_idx"] = users["user_id"].map(user_to_idx)

    encoder = _make_one_hot_encoder()
    gender_encoded = encoder.fit_transform(users[["gender"]])
    gender_columns = [f"gender_{category}" for category in encoder.categories_[0]]
    gender_df = pd.DataFrame(gender_encoded, columns=gender_columns, index=users.index)

    user_scaler = MinMaxScaler()
    users[["age_scaled", "interaction_count_scaled", "mean_rating_scaled"]] = user_scaler.fit_transform(
        users[["age", "interaction_count", "mean_rating"]]
    )

    user_features = pd.concat(
        [
            users[
                [
                    "user_id",
                    "user_idx",
                    "age",
                    "occupation",
                    "zip_prefix",
                    "interaction_count",
                    "mean_rating",
                    "age_scaled",
                    "interaction_count_scaled",
                    "mean_rating_scaled",
                ]
            ].reset_index(drop=True),
            gender_df.reset_index(drop=True),
        ],
        axis=1,
    )

    movies = movies[movies["movie_id"].isin(unique_movie_ids)].copy()
    movies = movies.merge(item_stats, on="movie_id", how="inner")
    movies["item_idx"] = movies["movie_id"].map(movie_to_idx)

    genre_binarizer = MultiLabelBinarizer()
    genre_matrix = genre_binarizer.fit_transform(movies["genres_list"])
    genre_columns = [f"genre_{_slugify_genre_name(genre)}" for genre in genre_binarizer.classes_]
    genre_df = pd.DataFrame(genre_matrix, columns=genre_columns, index=movies.index)

    item_scaler = MinMaxScaler()
    movies[["release_year_scaled", "interaction_count_scaled", "mean_rating_scaled"]] = item_scaler.fit_transform(
        movies[["release_year", "interaction_count", "mean_rating"]]
    )

    item_features = pd.concat(
        [
            movies[
                [
                    "movie_id",
                    "item_idx",
                    "title",
                    "release_year",
                    "interaction_count",
                    "mean_rating",
                    "release_year_scaled",
                    "interaction_count_scaled",
                    "mean_rating_scaled",
                ]
            ].reset_index(drop=True),
            genre_df.reset_index(drop=True),
        ],
        axis=1,
    )

    interactions = ratings.sort_values(["user_id", "timestamp", "movie_id"]).reset_index(drop=True)

    return {
        "interactions": interactions,
        "user_features": user_features.sort_values("user_idx").reset_index(drop=True),
        "item_features": item_features.sort_values("item_idx").reset_index(drop=True),
        "metadata": {
            "n_users": len(unique_user_ids),
            "n_items": len(unique_movie_ids),
            "genre_columns": genre_columns,
            "gender_columns": gender_columns,
        },
    }


def temporal_leave_last_k_out(
    interactions: pd.DataFrame,
    validation_k: int = 1,
    test_k: int = 1,
) -> dict[str, pd.DataFrame]:
    if validation_k < 0 or test_k < 0:
        raise ValueError("validation_k e test_k precisam ser nao negativos.")

    ordered = interactions.sort_values(["user_id", "timestamp", "movie_id"]).copy()
    user_sizes = ordered.groupby("user_id")["movie_id"].transform("size")
    min_required = validation_k + test_k + 1

    if (user_sizes < min_required).any():
        raise ValueError(
            "Cada usuario precisa ter pelo menos validation_k + test_k + 1 interacoes."
        )

    ordered["reverse_rank"] = ordered.groupby("user_id").cumcount(ascending=False) + 1

    test_mask = ordered["reverse_rank"] <= test_k
    validation_mask = (ordered["reverse_rank"] > test_k) & (
        ordered["reverse_rank"] <= test_k + validation_k
    )
    train_mask = ordered["reverse_rank"] > test_k + validation_k

    split_columns = [column for column in ordered.columns if column != "reverse_rank"]
    splits = {
        "train": ordered.loc[train_mask, split_columns].reset_index(drop=True),
        "validation": ordered.loc[validation_mask, split_columns].reset_index(drop=True),
        "test": ordered.loc[test_mask, split_columns].reset_index(drop=True),
    }

    return splits


def persist_processed_artifacts(
    feature_tables: dict[str, pd.DataFrame | dict],
    splits: dict[str, pd.DataFrame],
    output_dir: str | Path,
    quality_report: dict,
    seed: int = GLOBAL_SEED,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interactions = feature_tables["interactions"]
    user_features = feature_tables["user_features"]
    item_features = feature_tables["item_features"]
    metadata = feature_tables["metadata"]

    interactions.to_csv(output_dir / "interactions_full.csv", index=False)
    user_features.to_csv(output_dir / "user_features.csv", index=False)
    item_features.to_csv(output_dir / "item_features.csv", index=False)

    for split_name, split_df in splits.items():
        split_df.to_csv(output_dir / f"{split_name}_interactions.csv", index=False)

    artifact_summary = {
        "seed": seed,
        "n_interactions_full": int(len(interactions)),
        "n_train": int(len(splits["train"])),
        "n_validation": int(len(splits["validation"])),
        "n_test": int(len(splits["test"])),
        "n_users": int(metadata["n_users"]),
        "n_items": int(metadata["n_items"]),
        "genre_columns": metadata["genre_columns"],
        "gender_columns": metadata["gender_columns"],
        "quality_report": quality_report,
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(artifact_summary, metadata_file, indent=2, ensure_ascii=False)

    return artifact_summary


def run_pipeline(
    raw_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed/movielens",
    seed: int = GLOBAL_SEED,
) -> dict:
    set_global_seed(seed)

    bundle = load_movielens_data(raw_dir)
    raw_profile = profile_raw_data(bundle)
    clean_bundle, quality_report = clean_movielens_data(bundle)
    feature_tables = build_feature_tables(clean_bundle)
    splits = temporal_leave_last_k_out(feature_tables["interactions"])
    artifact_summary = persist_processed_artifacts(
        feature_tables=feature_tables,
        splits=splits,
        output_dir=output_dir,
        quality_report={
            "raw_profile": raw_profile,
            "cleaning": quality_report,
        },
        seed=seed,
    )

    return {
        "raw_profile": raw_profile,
        "clean_bundle": clean_bundle,
        "feature_tables": feature_tables,
        "splits": splits,
        "artifact_summary": artifact_summary,
    }
