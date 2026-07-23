"""Script para avaliação do modelo neural comparando com baselines."""

import json
from pathlib import Path

import pandas as pd
import torch

from src.config import settings
from src.evaluation.recommender_evaluation import run_benchmark
from src.models.baselines import BaseRecommender
from src.models.ncf import NCF


class NCFWrapper(BaseRecommender):
    """Wrapper para adaptar o modelo PyTorch para a interface do BaseRecommender."""

    def __init__(self, model: NCF) -> None:
        """Inicializa o wrapper.

        Args:
            model: O modelo NCF já treinado.
        """
        self.model = model
        self.model.eval()
        self.model_name = "ncf_pytorch"
        self.user_id_to_idx: dict[int, int] = {}
        self.movie_id_to_idx: dict[int, int] = {}
        self.idx_to_movie_id: dict[int, int] = {}

    def fit(
        self,
        interactions: pd.DataFrame,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
    ) -> "NCFWrapper":
        """Prepara os mapeamentos de IDs."""
        self.user_id_to_idx = dict(
            zip(
                user_features["user_id"].astype(int).tolist(),
                user_features["user_idx"].astype(int).tolist(),
            )
        )
        self.movie_id_to_idx = dict(
            zip(
                item_features["movie_id"].astype(int).tolist(),
                item_features["item_idx"].astype(int).tolist(),
            )
        )
        self.idx_to_movie_id = {
            item_idx: movie_id for movie_id, item_idx in self.movie_id_to_idx.items()
        }
        return self

    def recommend(
        self,
        user_id: int,
        candidate_item_ids: list[int],
        k: int,
    ) -> list[int]:
        """Prediz scores e ranqueia o top-K."""
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

        user_tensor = torch.full((len(candidate_indices),), user_idx, dtype=torch.long)
        item_tensor = torch.tensor(candidate_indices, dtype=torch.long)

        with torch.no_grad():
            scores = self.model(user_tensor, item_tensor).numpy()

        import numpy as np

        order = np.argsort(-scores, kind="stable")
        ranked_indices = [candidate_indices[pos] for pos in order[:k]]

        return [self.idx_to_movie_id[idx] for idx in ranked_indices]


def main():
    """Executa a avaliação do modelo neural comparando com baselines."""
    print("Carregando metadados...")
    processed_dir = Path(settings.data_dir) / "processed" / "movielens"
    test_df = pd.read_csv(processed_dir / "test_interactions.csv")

    with open(processed_dir / "metadata.json") as f:
        metadata = json.load(f)

    num_users_meta = metadata.get("num_users")
    num_users = (
        int(num_users_meta)
        if num_users_meta is not None
        else int(test_df["user_idx"].max() + 1)
    )

    num_items_meta = metadata.get("num_items")
    num_items = (
        int(num_items_meta)
        if num_items_meta is not None
        else int(test_df["item_idx"].max() + 1)
    )

    print("Carregando modelo treinado...")
    model = NCF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=settings.model.embedding_dim,
    )

    checkpoint_path = Path(settings.models_dir) / "ncf_best.pth"
    if not checkpoint_path.exists():
        print("Erro: Checkpoint não encontrado. Execute o treinamento primeiro.")
        return

    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    wrapped_model = NCFWrapper(model)

    print("Iniciando avaliação do modelo neural...")

    # Executa o mesmo benchmark dos baselines para avaliação justa!
    results = run_benchmark(
        recommenders=[wrapped_model],
        processed_dir=processed_dir,
        output_dir=processed_dir / "evaluation" / "neural",
        k=10,
    )

    print("\nResultados do NCF:")
    print(results["aggregate"])


if __name__ == "__main__":
    main()
