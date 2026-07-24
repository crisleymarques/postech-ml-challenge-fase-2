from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

try:
    from sklearn.neighbors import NearestNeighbors

    SKLEARN_MODELING_AVAILABLE = True
except ModuleNotFoundError:
    NearestNeighbors = None
    SKLEARN_MODELING_AVAILABLE = False


class BaseRecommender(ABC):
    """Interface comum para modelos de recomendação avaliados no projeto."""

    model_name: str

    @abstractmethod
    def fit(
        self,
        interactions: pd.DataFrame,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
    ) -> BaseRecommender:
        """Treina o modelo de recomendação com base nas interações."""
        raise NotImplementedError

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        candidate_item_ids: list[int],
        k: int,
    ) -> list[int]:
        """Gera um ranking top-K para o usuário a partir dos candidatos."""
        raise NotImplementedError


class PopularityRecommender(BaseRecommender):
    """Baseline simples baseado na popularidade histórica dos itens."""

    def __init__(self) -> None:
        """Inicializa o modelo de recomendação por popularidade."""
        self.model_name = "popularity"
        self.rankings: list[int] = []

    def fit(
        self,
        interactions: pd.DataFrame,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
    ) -> PopularityRecommender:
        """Calcula a popularidade dos itens e armazena o ranking global."""
        del user_features

        popularity = interactions.groupby("movie_id", as_index=False).agg(
            interaction_count=("user_id", "size"),
            mean_rating=("rating", "mean"),
        )
        popularity = popularity.sort_values(
            by=["interaction_count", "mean_rating", "movie_id"],
            ascending=[False, False, True],
        )

        seen_items = list(popularity["movie_id"])
        cold_items = item_features.loc[
            ~item_features["movie_id"].isin(seen_items), ["movie_id"]
        ].copy()
        cold_items["interaction_count"] = 0
        cold_items["mean_rating"] = 0.0

        full_ranking = pd.concat([popularity, cold_items], ignore_index=True)
        full_ranking = full_ranking.sort_values(
            by=["interaction_count", "mean_rating", "movie_id"],
            ascending=[False, False, True],
        )
        self.rankings = full_ranking["movie_id"].astype(int).tolist()
        return self

    def recommend(
        self,
        user_id: int,
        candidate_item_ids: list[int],
        k: int,
    ) -> list[int]:
        """Recomenda os itens mais populares dentre os candidatos."""
        del user_id

        candidate_set = set(candidate_item_ids)
        recommendations: list[int] = []
        for movie_id in self.rankings:
            if movie_id in candidate_set:
                recommendations.append(movie_id)
            if len(recommendations) == k:
                break
        return recommendations


class ItemKNNRecommender(BaseRecommender):
    """Baseline com Scikit-Learn baseado em similaridade item-item via KNN."""

    def __init__(self, n_neighbors: int = 40) -> None:
        """Inicializa o modelo KNN baseado em itens com Scikit-Learn."""
        if not SKLEARN_MODELING_AVAILABLE:
            raise ModuleNotFoundError(
                "Scikit-Learn não está disponível para o baseline ItemKNNRecommender."
            )

        self.model_name = "item_knn_sklearn"
        self.n_neighbors = n_neighbors
        self.user_id_to_idx: dict[int, int] = {}
        self.movie_id_to_idx: dict[int, int] = {}
        self.idx_to_movie_id: dict[int, int] = {}
        self.user_item_matrix: csr_matrix | None = None
        self.item_similarity_matrix: csr_matrix | None = None
        self.popularity_scores: np.ndarray | None = None

    def fit(
        self,
        interactions: pd.DataFrame,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
    ) -> ItemKNNRecommender:
        """Constrói a matriz de similaridade item-item usando NearestNeighbors."""
        self.user_id_to_idx = {
            int(u): int(idx)
            for u, idx in zip(user_features["user_id"], user_features["user_idx"])
        }
        self.movie_id_to_idx = {
            int(m): int(idx)
            for m, idx in zip(item_features["movie_id"], item_features["item_idx"])
        }
        self.idx_to_movie_id = {
            item_idx: movie_id for movie_id, item_idx in self.movie_id_to_idx.items()
        }

        n_users = len(self.user_id_to_idx)
        n_items = len(self.movie_id_to_idx)
        rows = interactions["user_idx"].to_numpy(dtype=np.int32)
        cols = interactions["item_idx"].to_numpy(dtype=np.int32)
        values = (interactions["rating"].to_numpy(dtype=np.float32) / 5.0).astype(
            np.float32
        )

        self.user_item_matrix = cast(
            csr_matrix,
            coo_matrix((values, (rows, cols)), shape=(n_users, n_items)).tocsr(),
        )

        assert NearestNeighbors is not None
        knn = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=min(self.n_neighbors + 1, n_items),
        )
        assert self.user_item_matrix is not None
        item_user_matrix = self.user_item_matrix.T.tocsr()
        knn.fit(item_user_matrix)
        distances, indices = knn.kneighbors(item_user_matrix, return_distance=True)

        similarity_data: list[float] = []
        similarity_rows: list[int] = []
        similarity_cols: list[int] = []

        for item_idx in range(n_items):
            for neighbor_idx, distance in zip(indices[item_idx], distances[item_idx]):
                if neighbor_idx == item_idx:
                    continue
                similarity = max(0.0, 1.0 - float(distance))
                if similarity > 0:
                    similarity_rows.append(item_idx)
                    similarity_cols.append(int(neighbor_idx))
                    similarity_data.append(similarity)

        self.item_similarity_matrix = cast(
            csr_matrix,
            coo_matrix(
                (
                    np.asarray(similarity_data, dtype=np.float32),
                    (similarity_rows, similarity_cols),
                ),
                shape=(n_items, n_items),
            ).tocsr(),
        )

        popularity_counts = (
            interactions.groupby("item_idx")["user_id"]
            .size()
            .reindex(range(n_items), fill_value=0)
            .to_numpy(dtype=np.float32)
        )
        max_count = popularity_counts.max() if len(popularity_counts) else 1.0
        if max_count == 0:
            max_count = 1.0
        self.popularity_scores = cast(np.ndarray, popularity_counts / max_count)
        return self

    def recommend(
        self,
        user_id: int,
        candidate_item_ids: list[int],
        k: int,
    ) -> list[int]:
        """Gera recomendações baseadas na similaridade com o perfil do usuário."""
        if (
            self.user_item_matrix is None
            or self.item_similarity_matrix is None
            or self.popularity_scores is None
        ):
            raise RuntimeError(
                "O modelo precisa ser treinado antes de gerar recomendações."
            )

        user_idx = self.user_id_to_idx.get(user_id)
        if user_idx is None:
            return candidate_item_ids[:k]

        candidate_indices = [
            self.movie_id_to_idx[item_id]
            for item_id in candidate_item_ids
            if item_id in self.movie_id_to_idx
        ]
        if not candidate_indices:
            return []

        user_profile = self.user_item_matrix.getrow(user_idx)
        pop_scores = self.popularity_scores
        if user_profile.nnz == 0:
            popularity_order = sorted(
                candidate_indices,
                key=lambda idx: (pop_scores[idx], -idx),
                reverse=True,
            )
            return [self.idx_to_movie_id[idx] for idx in popularity_order[:k]]

        score_vector = (user_profile @ self.item_similarity_matrix).toarray().ravel()
        score_vector = score_vector + (1e-6 * self.popularity_scores)
        if user_profile.nnz > 0:
            score_vector[user_profile.indices] = -np.inf

        candidate_scores = score_vector[candidate_indices]
        order = np.argsort(-candidate_scores, kind="stable")
        ranked_indices = [candidate_indices[position] for position in order[:k]]
        return [self.idx_to_movie_id[idx] for idx in ranked_indices]


def ensure_output_dir(path: str | Path) -> Path:
    """Garante que o diretório de saída exista, criando-o se necessário."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
