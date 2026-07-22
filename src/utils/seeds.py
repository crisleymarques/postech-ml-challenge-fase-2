"""Utilitários para garantia de reprodutibilidade no treinamento."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Fixa as sementes globais para reprodutibilidade.

    Aplica a seed fornecida nos geradores de números pseudo-aleatórios do
    Python, NumPy e PyTorch, e configura operações determinísticas.

    Args:
        seed: Valor inteiro para a semente. Padrão é 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
