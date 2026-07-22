import torch

from src.models.ncf import NCF


def test_ncf_output_shape():
    """Verifica se o modelo NCF retorna o shape esperado (batch_size,)."""
    batch_size = 16
    num_users = 100
    num_items = 50
    embedding_dim = 16

    model = NCF(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim)

    # Cria batch simulado
    users = torch.randint(0, num_users, (batch_size,))
    items = torch.randint(0, num_items, (batch_size,))

    output = model(users, items)

    assert output.shape == (batch_size,)
    assert output.dtype == torch.float32


def test_ncf_different_layers():
    """Testa se o modelo suporta configuração diferente de hidden layers."""
    model = NCF(
        num_users=10,
        num_items=10,
        hidden_layers=[32, 16]
    )
    # 2 linear, 2 relu, 2 dropout
    assert len(model.mlp) == 6
