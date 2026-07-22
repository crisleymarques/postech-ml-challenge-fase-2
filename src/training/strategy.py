"""Implementação do Strategy Pattern para rotinas de treinamento neural."""

from abc import ABC, abstractmethod
from typing import Any

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class TrainingStrategy(ABC):
    """Interface base para estratégias de treinamento."""

    @abstractmethod
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, Any]:
        """Executa a rotina de treinamento e validação.

        Args:
            model: O modelo PyTorch a ser treinado.
            train_loader: DataLoader com os dados de treinamento.
            val_loader: DataLoader com os dados de validação.

        Returns:
            Dicionário com o histórico de métricas do treinamento.
        """
        pass


class NCFTrainingStrategy(TrainingStrategy):
    """Estratégia de treinamento específica para o modelo NCF.

    Inclui loop de epochs, cálculo de loss, early stopping,
    checkpointing do melhor modelo e tracking via MLflow.
    """

    def __init__(
        self,
        epochs: int = 20,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 5,
        device: str = "cpu",
        checkpoint_path: str = "models/best_ncf.pth",
    ) -> None:
        """Inicializa a estratégia.

        Args:
            epochs: Número máximo de épocas.
            learning_rate: Taxa de aprendizado para o otimizador.
            early_stopping_patience: Paciência para parada precoce.
            device: Dispositivo para alocar os tensores ('cpu' ou 'cuda').
            checkpoint_path: Caminho para salvar o melhor modelo.
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = early_stopping_patience
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, Any]:
        """Executa o loop de treinamento e validação."""
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        # Usamos BCEWithLogitsLoss porque as interações são implícitas (0 ou 1)
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
            val_loss = self._validate_epoch(model, val_loader, criterion)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            print(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(model)
                print("Melhor modelo salvo!")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}!")
                    break

        return history

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Executa uma época de treinamento."""
        model.train()
        total_loss = 0.0

        for user, item, target in tqdm(loader, desc="Training", leave=False):
            user = user.to(self.device)
            item = item.to(self.device)
            target = target.to(self.device)

            optimizer.zero_grad()
            predictions = model(user, item)
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _validate_epoch(
        self, model: nn.Module, loader: DataLoader, criterion: nn.Module
    ) -> float:
        """Executa a validação do modelo."""
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for user, item, target in loader:
                user = user.to(self.device)
                item = item.to(self.device)
                target = target.to(self.device)

                predictions = model(user, item)
                loss = criterion(predictions, target)
                total_loss += loss.item()

        return total_loss / len(loader)

    def _save_checkpoint(self, model: nn.Module) -> None:
        """Salva os pesos do modelo no disco."""
        import os

        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), self.checkpoint_path)

    def load_checkpoint(self, model: nn.Module) -> nn.Module:
        """Carrega os pesos salvos para o modelo."""
        model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device)
        )  # noqa: E501
        return model
