from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.movielens_pipeline import (
    MovieLensBundle,
    build_feature_tables,
    clean_movielens_data,
    profile_raw_data,
    temporal_leave_last_k_out,
)


class MovieLensPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = MovieLensBundle(
            ratings=pd.DataFrame(
                [
                    {"user_id": 1, "movie_id": 10, "rating": 4, "timestamp": 100},
                    {"user_id": 1, "movie_id": 10, "rating": 5, "timestamp": 200},
                    {"user_id": 1, "movie_id": 11, "rating": 3, "timestamp": 300},
                    {"user_id": 1, "movie_id": 12, "rating": 2, "timestamp": 400},
                    {"user_id": 2, "movie_id": 10, "rating": 4, "timestamp": 150},
                    {"user_id": 2, "movie_id": 12, "rating": 5, "timestamp": 250},
                    {"user_id": 2, "movie_id": 13, "rating": 1, "timestamp": 350},
                    {"user_id": 2, "movie_id": 99, "rating": 4, "timestamp": 450},
                ]
            ),
            users=pd.DataFrame(
                [
                    {
                        "user_id": 1,
                        "gender": "F",
                        "age": 25,
                        "occupation": 1,
                        "zip_code": "12345",
                    },
                    {
                        "user_id": 2,
                        "gender": "M",
                        "age": 35,
                        "occupation": 2,
                        "zip_code": "54321",
                    },
                ]
            ),
            movies=pd.DataFrame(
                [
                    {
                        "movie_id": 10,
                        "title": "Movie A (1999)",
                        "genres": "Action|Comedy",
                    },
                    {"movie_id": 11, "title": "Movie B (2001)", "genres": "Drama"},
                    {"movie_id": 12, "title": "Movie C (2002)", "genres": "Comedy"},
                    {"movie_id": 13, "title": "Movie D", "genres": "Thriller"},
                ]
            ),
        )

    def test_profile_raw_data_identifica_pares_duplicados_e_referencias_invalidas(
        self,
    ) -> None:
        profile = profile_raw_data(self.bundle)

        self.assertEqual(profile["duplicate_rows"]["ratings_user_movie_pairs"], 1)
        self.assertEqual(profile["invalid_references"]["ratings_without_movie"], 1)
        self.assertEqual(profile["summary"]["n_ratings"], 8)

    def test_clean_movielens_remove_interacao_invalida_e_mantem_rating_mais_recente(
        self,
    ) -> None:
        clean_bundle, quality_report = clean_movielens_data(self.bundle)

        self.assertEqual(len(clean_bundle.ratings), 6)
        self.assertFalse((clean_bundle.ratings["movie_id"] == 99).any())

        user_one_movie_ten = clean_bundle.ratings[
            (clean_bundle.ratings["user_id"] == 1)
            & (clean_bundle.ratings["movie_id"] == 10)
        ]
        self.assertEqual(len(user_one_movie_ten), 1)
        self.assertEqual(user_one_movie_ten.iloc[0]["rating"], 5)
        self.assertIn("release_year", clean_bundle.movies.columns)
        self.assertEqual(quality_report["rows_removed"]["ratings"], 2)

    def test_build_feature_tables_gera_indices_e_features_numericas(self) -> None:
        clean_bundle, _ = clean_movielens_data(self.bundle)
        feature_tables = build_feature_tables(clean_bundle)

        self.assertIn("user_idx", feature_tables["interactions"].columns)
        self.assertIn("item_idx", feature_tables["interactions"].columns)
        self.assertIn("age_scaled", feature_tables["user_features"].columns)
        self.assertTrue(
            any(
                column.startswith("genre_")
                for column in feature_tables["item_features"].columns
            )
        )
        self.assertEqual(feature_tables["metadata"]["n_users"], 2)

    def test_temporal_leave_last_k_out_separa_sem_vazamento_temporal(self) -> None:
        clean_bundle, _ = clean_movielens_data(self.bundle)
        feature_tables = build_feature_tables(clean_bundle)
        splits = temporal_leave_last_k_out(
            feature_tables["interactions"], validation_k=1, test_k=1
        )

        for user_id in feature_tables["interactions"]["user_id"].unique():
            user_train = splits["train"].loc[
                splits["train"]["user_id"] == user_id, "timestamp"
            ]
            user_validation = splits["validation"].loc[
                splits["validation"]["user_id"] == user_id, "timestamp"
            ]
            user_test = splits["test"].loc[
                splits["test"]["user_id"] == user_id, "timestamp"
            ]

            self.assertLess(user_train.max(), user_validation.min())
            self.assertLess(user_validation.max(), user_test.min())


if __name__ == "__main__":
    unittest.main()
