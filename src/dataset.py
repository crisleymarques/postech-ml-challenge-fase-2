import torch
from torch.utils.data import Dataset


class RecommendationDataset(Dataset):
    """Dataset para pares de interação Usuário-Item."""

    def __init__(self, user_ids: torch.Tensor, item_ids: torch.Tensor, labels: torch.Tensor):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "label": self.labels[idx],
        }