"""Módulo de configuração para o ML challenge.

Este módulo usa o pydantic-settings para carregar a configuração a partir de
variáveis de ambiente e de um arquivo .env opcional, fornecendo um objeto de
configurações centralizado.
"""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseModel):
    """Hiperparâmetros do modelo neural de recomendação."""

    learning_rate: float = Field(
        default=0.001, description="Taxa de aprendizado (learning rate)."
    )
    batch_size: int = Field(default=256, description="Tamanho do batch.")
    embedding_dim: int = Field(
        default=32, description="Dimensão dos embeddings (usuários e itens)."
    )
    epochs: int = Field(default=20, description="Número máximo de épocas.")
    early_stopping_patience: int = Field(
        default=5, description="Paciência do early stopping."
    )


class Settings(BaseSettings):
    """Configurações centralizadas carregadas do ambiente ou de um arquivo .env.

    Attributes:
        environment: O estágio da aplicação (development, production, etc).
        mlflow_tracking_uri: O URI de tracking para o servidor MLflow.
        mlflow_experiment_name: O nome do experimento para rastrear os runs.
        dvc_remote_path: Caminho opcional para o armazenamento remoto do DVC.
        data_dir: Diretório local de dados.
        models_dir: Diretório local de modelos.
        configs_dir: Diretório local de configurações.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = Field(
        default="development",
        description="Estágio do ambiente",
    )
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="URI de tracking do servidor MLflow",
    )
    mlflow_experiment_name: str = Field(
        default="postech-challenge",
        description="Nome do experimento do MLflow",
    )
    dvc_remote_path: str | None = Field(
        default=None,
        description="Caminho opcional para o armazenamento remoto do DVC",
    )
    data_dir: str = Field(
        default="data",
        description="Diretório local de dados",
    )
    models_dir: str = Field(
        default="models",
        description="Diretório local de modelos",
    )
    configs_dir: str = Field(
        default="configs",
        description="Diretório local de configurações",
    )
    model: ModelSettings = Field(default_factory=ModelSettings)


# Instancia as configurações para uso global no projeto
settings = Settings()
