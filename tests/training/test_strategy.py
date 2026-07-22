from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.ncf import NCF
from src.training.strategy import NCFTrainingStrategy


@pytest.fixture
def dummy_data():
    """Gera dados falsos para teste."""
    users = torch.randint(0, 10, (100,))
    items = torch.randint(0, 10, (100,))
    targets = torch.randint(0, 2, (100,)).float()

    dataset = TensorDataset(users, items, targets)
    loader = DataLoader(dataset, batch_size=10)
    return loader


@pytest.fixture
def model():  # noqa: D103
    return NCF(num_users=10, num_items=10, embedding_dim=8)


@patch("mlflow.log_metric")
def test_early_stopping(mock_log, model, dummy_data, tmp_path):
    """Testa se o early stopping interrompe o treinamento corretamente."""
    checkpoint = tmp_path / "model.pth"

    strategy = NCFTrainingStrategy(
        epochs=10, early_stopping_patience=2, checkpoint_path=str(checkpoint)
    )

    # Força a validação a piorar progressivamente
    strategy._validate_epoch = MagicMock(side_effect=[0.5, 0.6, 0.7, 0.8])

    history = strategy.train(model, dummy_data, dummy_data)

    # Deve parar na época 3 (índice 2) devido a patience = 2
    assert len(history["val_loss"]) == 3
    assert checkpoint.exists()


def test_checkpoint_save_load(model, tmp_path):
    """Testa o salvamento e carregamento dos pesos do modelo."""
    checkpoint = tmp_path / "model.pth"
    strategy = NCFTrainingStrategy(checkpoint_path=str(checkpoint))

    # Salva
    strategy._save_checkpoint(model)
    assert checkpoint.exists()

    # Modifica o modelo
    with torch.no_grad():
        model.user_embedding.weight.fill_(0)

    # Carrega e verifica se restaurou
    strategy.load_checkpoint(model)
    assert not torch.allclose(
        model.user_embedding.weight, torch.zeros_like(model.user_embedding.weight)
    )  # noqa: E501
