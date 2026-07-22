"""Módulo com a arquitetura Neural Collaborative Filtering (NCF) em PyTorch."""

import torch
from torch import nn


class NCF(nn.Module):
    """Modelo Neural Collaborative Filtering (NCF).

    Este modelo combina duas matrizes de embeddings latentes (usuários e itens)
    e as concatena para passar por uma série de camadas lineares densas (MLP).
    Utilizado para aprender padrões complexos de interação em recomendação implícita
    ou predição de ratings.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        hidden_layers: list[int] | None = None,
        dropout_rate: float = 0.2,
    ) -> None:
        """Inicializa as camadas do NCF.

        Args:
            num_users: Número total de usuários únicos (tamanho do vocabulário).
            num_items: Número total de itens únicos (tamanho do vocabulário).
            embedding_dim: Dimensão dos vetores de embedding. Padrão é 32.
            hidden_layers: Lista com os tamanhos das camadas ocultas do MLP.
                Se None, o padrão é [64, 32, 16].
            dropout_rate: Taxa de dropout para as camadas intermediárias.
        """
        super().__init__()

        # Camadas de Embedding
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users, embedding_dim=embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items, embedding_dim=embedding_dim
        )

        # Configura as camadas ocultas (MLP)
        if hidden_layers is None:
            hidden_layers = [64, 32, 16]

        # A primeira camada MLP recebe a concatenação dos dois embeddings
        mlp_input_dim = embedding_dim * 2

        layers = []
        in_dim = mlp_input_dim

        for out_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

        # Camada final de saída
        # Como as interações podem ser 1.0/0.0, usamos Sigmoid em inferência,
        # mas aqui retornamos os logits e aplicamos BCEWithLogitsLoss no treino.
        self.output_layer = nn.Linear(in_dim, 1)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:  # noqa: E501
        """Executa a passagem para frente (forward pass) do modelo.

        Args:
            user_indices: Tensor 1D com os IDs numéricos dos usuários.
            item_indices: Tensor 1D com os IDs numéricos dos itens.

        Returns:
            Tensor 1D contendo a predição para o par usuário-item.
        """
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)

        # Concatena lado a lado (batch_size, embedding_dim * 2)
        concat_embed = torch.cat([user_embed, item_embed], dim=-1)

        # Passa pelo MLP
        mlp_output = self.mlp(concat_embed)

        # Calcula a predição final
        prediction = self.output_layer(mlp_output)

        return prediction.squeeze(-1)
