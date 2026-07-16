"""Testes unitários para as estratégias de pré-processamento de dados."""

import numpy as np

from src.preprocessing import (
    DataPreprocessor,
    MinMaxScalerStrategy,
    StandardScalerStrategy,
)


def test_standard_scaler_strategy() -> None:
    """Testa a estratégia StandardScalerStrategy em dados sintéticos."""
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    strategy = StandardScalerStrategy()
    strategy.fit(data)
    transformed = strategy.transform(data)

    # Garante que a média seja próxima de 0 e o desvio padrão (std) próximo de 1
    assert np.allclose(transformed.mean(axis=0), [0.0, 0.0])
    assert np.allclose(transformed.std(axis=0), [1.0, 1.0])


def test_min_max_scaler_strategy() -> None:
    """Testa a estratégia MinMaxScalerStrategy em dados sintéticos."""
    data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    strategy = MinMaxScalerStrategy()
    strategy.fit(data)
    transformed = strategy.transform(data)

    # Garante que o valor mínimo seja 0 e o valor máximo seja 1
    assert np.allclose(transformed.min(axis=0), [0.0, 0.0])
    assert np.allclose(transformed.max(axis=0), [1.0, 1.0])


def test_data_preprocessor_context() -> None:
    """Testa o contexto DataPreprocessor e a troca dinâmica de estratégias."""
    data = np.array([[0.0], [5.0], [10.0]])

    # Inicializa com o MinMaxScaler
    preprocessor = DataPreprocessor(MinMaxScalerStrategy())
    res_minmax = preprocessor.fit_transform(data)
    assert np.allclose(res_minmax, [[0.0], [0.5], [1.0]])

    # Altera para o StandardScaler dinamicamente
    preprocessor.set_strategy(StandardScalerStrategy())
    res_std = preprocessor.fit_transform(data)
    assert np.allclose(res_std.mean(), 0.0)
