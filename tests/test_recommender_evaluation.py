from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.recommender_evaluation import (
    catalog_coverage_at_k,
    evaluate_fitted_recommender,
    evaluate_recommender,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.models.baselines import ItemKNNRecommender, PopularityRecommender
from src.models.model_persistence import load_trained_recommenders, save_trained_recommenders


class RecommenderEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "movie_id": 101,
                    "rating": 5.0,
                    "user_idx": 0,
                    "item_idx": 0,
                },
                {
                    "user_id": 1,
                    "movie_id": 102,
                    "rating": 4.0,
                    "user_idx": 0,
                    "item_idx": 1,
                },
                {
                    "user_id": 2,
                    "movie_id": 101,
                    "rating": 4.0,
                    "user_idx": 1,
                    "item_idx": 0,
                },
                {
                    "user_id": 2,
                    "movie_id": 103,
                    "rating": 5.0,
                    "user_idx": 1,
                    "item_idx": 2,
                },
                {
                    "user_id": 3,
                    "movie_id": 102,
                    "rating": 4.0,
                    "user_idx": 2,
                    "item_idx": 1,
                },
                {
                    "user_id": 3,
                    "movie_id": 104,
                    "rating": 5.0,
                    "user_idx": 2,
                    "item_idx": 3,
                },
            ]
        )
        self.test = pd.DataFrame(
            [
                {"user_id": 1, "movie_id": 103},
                {"user_id": 2, "movie_id": 104},
                {"user_id": 3, "movie_id": 101},
            ]
        )
        self.user_features = pd.DataFrame(
            [
                {"user_id": 1, "user_idx": 0},
                {"user_id": 2, "user_idx": 1},
                {"user_id": 3, "user_idx": 2},
            ]
        )
        self.item_features = pd.DataFrame(
            [
                {"movie_id": 101, "item_idx": 0},
                {"movie_id": 102, "item_idx": 1},
                {"movie_id": 103, "item_idx": 2},
                {"movie_id": 104, "item_idx": 3},
            ]
        )

    def test_metricas_de_ranking_funcionam_com_um_item_relevante(self) -> None:
        recommended = [103, 104, 101]
        relevant = {103}

        self.assertAlmostEqual(precision_at_k(recommended, relevant, 3), 1 / 3)
        self.assertAlmostEqual(recall_at_k(recommended, relevant, 3), 1.0)
        self.assertAlmostEqual(hit_rate_at_k(recommended, relevant, 3), 1.0)
        self.assertAlmostEqual(ndcg_at_k(recommended, relevant, 3), 1.0)
        self.assertAlmostEqual(mrr_at_k(recommended, relevant, 3), 1.0)

    def test_catalog_coverage_mede_diversidade_das_recomendacoes(self) -> None:
        coverage = catalog_coverage_at_k(
            recommendations_by_user={1: [101, 102], 2: [102, 103]},
            all_item_ids=[101, 102, 103, 104],
            k=2,
        )
        self.assertAlmostEqual(coverage, 0.75)

    def test_popularity_recommender_retorna_itens_nao_vistos(self) -> None:
        model = PopularityRecommender().fit(
            self.train, self.user_features, self.item_features
        )
        recommendations = model.recommend(user_id=1, candidate_item_ids=[103, 104], k=2)
        self.assertEqual(recommendations, [103, 104])

    def test_item_knn_recommender_exclui_itens_vistos_e_retorna_top_k(self) -> None:
        model = ItemKNNRecommender(n_neighbors=2).fit(
            self.train, self.user_features, self.item_features
        )
        recommendations = model.recommend(user_id=1, candidate_item_ids=[103, 104], k=2)
        self.assertEqual(len(recommendations), 2)
        self.assertNotIn(101, recommendations)
        self.assertNotIn(102, recommendations)

    def test_evaluate_recommender_gera_metricas_agregadas_e_detalhadas(self) -> None:
        aggregate, detailed = evaluate_recommender(
            recommender=PopularityRecommender(),
            fit_interactions=self.train,
            test_interactions=self.test,
            user_features=self.user_features,
            item_features=self.item_features,
            k=2,
        )

        self.assertEqual(aggregate["model_name"], "popularity")
        self.assertEqual(aggregate["n_users_evaluated"], 3)
        self.assertIn("catalog_coverage_at_k", aggregate)
        self.assertEqual(len(detailed), 3)
        self.assertIn("recommended_items", detailed.columns)

    def test_evaluate_fitted_recommender_reutiliza_modelo_pre_treinado(self) -> None:
        model = PopularityRecommender().fit(self.train, self.user_features, self.item_features)

        aggregate, detailed = evaluate_fitted_recommender(
            recommender=model,
            fit_interactions=self.train,
            test_interactions=self.test,
            item_features=self.item_features,
            k=2,
        )

        self.assertEqual(aggregate["model_name"], "popularity")
        self.assertEqual(aggregate["n_users_evaluated"], 3)
        self.assertEqual(len(detailed), 3)

    def test_model_persistence_salva_e_recarrega_recomendadores_treinados(self) -> None:
        recommenders = [
            PopularityRecommender().fit(self.train, self.user_features, self.item_features),
            ItemKNNRecommender(n_neighbors=2).fit(self.train, self.user_features, self.item_features),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            payload = save_trained_recommenders(
                recommenders=recommenders,
                output_dir=temp_dir,
                training_summary={"fit_split": "train"},
            )
            loaded_recommenders, training_summary = load_trained_recommenders(temp_dir)

        self.assertEqual(len(payload["models"]), 2)
        self.assertEqual(training_summary["fit_split"], "train")
        self.assertEqual([model.model_name for model in loaded_recommenders], ["popularity", "item_knn_sklearn"])


if __name__ == "__main__":
    unittest.main()
