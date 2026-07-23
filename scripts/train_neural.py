"""Script de treinamento do modelo neural."""

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd
from torch.utils.data import DataLoader

from src.config import settings
from src.data.recsys_dataset import InteractionDataset
from src.models.ncf import NCF
from src.training.strategy import NCFTrainingStrategy
from src.utils.seeds import seed_everything


def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Carrega os dados processados e metadados."""
    processed_dir = Path(data_dir) / "processed" / "movielens"

    train_df = pd.read_csv(processed_dir / "train_interactions.csv")
    val_df = pd.read_csv(processed_dir / "validation_interactions.csv")
    test_df = pd.read_csv(processed_dir / "test_interactions.csv")

    with open(processed_dir / "metadata.json") as f:
        metadata = json.load(f)

    return train_df, val_df, test_df, metadata


def main():
    """Executa o treinamento do modelo neural NCF."""
    parser = argparse.ArgumentParser(description="Treina o modelo NCF.")
    parser.add_argument("--seed", type=int, default=42, help="Seed global.")
    args = parser.parse_args()

    print("Configurando reprodutibilidade...")
    seed_everything(args.seed)

    print("Carregando dados processados...")
    train_df, val_df, test_df, metadata = load_data(settings.data_dir)

    # Extrai tamanhos do vocabulário do metadata original
    # (Se não estiver no metadata, calcula do tamanho máximo)
    num_users_meta = metadata.get("num_users")
    num_users = (
        int(num_users_meta)
        if num_users_meta is not None
        else int(max(train_df["user_idx"].max(), val_df["user_idx"].max()) + 1)
    )

    num_items_meta = metadata.get("num_items")
    num_items = (
        int(num_items_meta)
        if num_items_meta is not None
        else int(max(train_df["item_idx"].max(), val_df["item_idx"].max()) + 1)
    )

    print(f"Num Users: {num_users}, Num Items: {num_items}")

    print("Criando datasets e dataloaders...")
    train_dataset = InteractionDataset(train_df, interaction_col="interaction")
    val_dataset = InteractionDataset(val_df, interaction_col="interaction")

    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.model.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.model.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("Inicializando a arquitetura NCF...")
    model = NCF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=settings.model.embedding_dim,
    )

    checkpoint_path = Path(settings.models_dir) / "ncf_best.pth"

    print("Configurando a estratégia de treinamento (NCF Strategy)...")
    strategy = NCFTrainingStrategy(
        epochs=settings.model.epochs,
        learning_rate=settings.model.learning_rate,
        early_stopping_patience=settings.model.early_stopping_patience,
        checkpoint_path=str(checkpoint_path),
        device="cpu",
    )

    print("Iniciando rotina de treinamento com MLflow...")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="ncf_training"):
        mlflow.log_param("seed", args.seed)
        mlflow.log_params(settings.model.model_dump())
        mlflow.log_param("num_users", num_users)
        mlflow.log_param("num_items", num_items)

        _ = strategy.train(model, train_loader, val_loader)

        # O melhor modelo foi salvo pelo Early Stopping. Rastrear no MLflow.
        if checkpoint_path.exists():
            mlflow.log_artifact(str(checkpoint_path), artifact_path="models")

        print("Treinamento finalizado com sucesso!")


if __name__ == "__main__":
    main()
