"""Dataset customizado para PyTorch focado em Sistemas de Recomendação."""

import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """Dataset para carregar interações usuário-item no PyTorch.

    Recebe um DataFrame pandas com colunas específicas de usuário, item
    e interações (opcional). Converte os dados em tensores otimizados.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        user_col: str = "user_id_idx",
        item_col: str = "item_id_idx",
        interaction_col: str | None = "interaction",
    ) -> None:
        """Inicializa o Dataset convertendo as colunas para tensores PyTorch.

        Args:
            df: DataFrame contendo as interações.
            user_col: Nome da coluna que representa o ID (índice mapeado) do usuário.
            item_col: Nome da coluna que representa o ID (índice mapeado) do item.
            interaction_col: Coluna alvo opcional. Se None, o dataset não retorna labels.
        """  # noqa: E501
        if user_col not in df.columns or item_col not in df.columns:
            raise ValueError(
                f"As colunas '{user_col}' e '{item_col}' devem existir no DataFrame."
            )

        self.users = torch.tensor(df[user_col].values, dtype=torch.long)
        self.items = torch.tensor(df[item_col].values, dtype=torch.long)
        self.has_interactions = False

        self.interactions: torch.Tensor | None = None
        if interaction_col is not None and interaction_col in df.columns:
            self.interactions = torch.tensor(
                df[interaction_col].values, dtype=torch.float32
            )
            self.has_interactions = True

    def __len__(self) -> int:
        """Retorna o número total de exemplos no dataset."""
        return len(self.users)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # noqa: E501
        """Recupera a interação dado um índice.

        Args:
            index: Índice do registro.

        Returns:
            Tupla contendo (tensor_usuario, tensor_item, tensor_interacao).
            Caso interaction_col seja nulo, o terceiro elemento será um tensor vazio.
        """
        user = self.users[index]
        item = self.items[index]

        if self.interactions is not None:
            interaction = self.interactions[index]
        else:
            interaction = torch.tensor(0.0, dtype=torch.float32)

        return user, item, interaction
