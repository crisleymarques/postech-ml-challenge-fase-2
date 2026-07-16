"""Estratégias de pré-processamento de dados utilizando o padrão Strategy.

Este módulo fornece uma abstração e implementações concretas para estratégias de
redimensionamento (scaling) de dados usando o padrão de projeto Strategy.
"""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class PreprocessorStrategy(ABC):
    """Classe base abstrata para estratégias de pré-processamento de dados."""

    @abstractmethod
    def fit(self, data: np.ndarray) -> "PreprocessorStrategy":
        """Ajusta o pré-processador com os dados fornecidos.

        Args:
            data: Array numérico para calcular os parâmetros.

        Returns:
            A instância da estratégia ajustada.
        """
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transforma os dados utilizando os parâmetros ajustados.

        Args:
            data: Array numérico a ser transformado.

        Returns:
            O array numérico transformado.
        """
        pass


class StandardScalerStrategy(PreprocessorStrategy):
    """Estratégia de pré-processamento para média zero e variância unitária.

    Attributes:
        scaler: A instância interna do StandardScaler do Scikit-Learn.
    """

    def __init__(self) -> None:
        """Inicializa a estratégia standard scaler instanciando o scaler interno."""
        self.scaler = StandardScaler()

    def fit(self, data: np.ndarray) -> "StandardScalerStrategy":
        """Ajusta os parâmetros do standard scaler.

        Args:
            data: Array numpy de entrada.

        Returns:
            A própria instância (self).
        """
        self.scaler.fit(data)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transforma o array usando o escalonamento padrão (standard scaling).

        Args:
            data: Array numpy de entrada.

        Returns:
            Array numpy escalonado.
        """
        return self.scaler.transform(data)


class MinMaxScalerStrategy(PreprocessorStrategy):
    """Estratégia de pré-processamento com escala de intervalo definido (padrão 0-1).

    Attributes:
        scaler: A instância interna do MinMaxScaler do Scikit-Learn.
    """

    def __init__(self) -> None:
        """Inicializa a estratégia min-max scaler instanciando o scaler interno."""
        self.scaler = MinMaxScaler()

    def fit(self, data: np.ndarray) -> "MinMaxScalerStrategy":
        """Ajusta os parâmetros do min-max scaler.

        Args:
            data: Array numpy de entrada.

        Returns:
            A própria instância (self).
        """
        self.scaler.fit(data)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transforma o array de entrada usando o escalonamento min-max.

        Args:
            data: Array numpy de entrada.

        Returns:
            Array numpy escalonado.
        """
        return self.scaler.transform(data)


class DataPreprocessor:
    """Classe de contexto que executa a PreprocessorStrategy selecionada.

    Attributes:
        strategy: A implementação atual da PreprocessorStrategy.
    """

    def __init__(self, strategy: PreprocessorStrategy) -> None:
        """Inicializa o contexto com a estratégia escolhida.

        Args:
            strategy: Estratégia de pré-processamento a ser utilizada.
        """
        self.strategy = strategy

    def set_strategy(self, strategy: PreprocessorStrategy) -> None:
        """Altera dinamicamente a estratégia de pré-processamento.

        Args:
            strategy: Nova estratégia de pré-processamento.
        """
        self.strategy = strategy

    def fit(self, data: np.ndarray) -> "DataPreprocessor":
        """Ajusta a estratégia aos dados.

        Args:
            data: Array numpy de entrada.

        Returns:
            A própria instância (self).
        """
        self.strategy.fit(data)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transforma os dados utilizando a estratégia selecionada.

        Args:
            data: Array numpy de entrada.

        Returns:
            Array numpy pré-processado.
        """
        return self.strategy.transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Ajusta e transforma os dados em uma única etapa.

        Args:
            data: Array numpy de entrada.

        Returns:
            Array numpy pré-processado.
        """
        return self.strategy.fit(data).transform(data)
