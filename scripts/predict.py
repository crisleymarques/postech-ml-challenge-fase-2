import sys
from pathlib import Path
import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings


def main():
    # 1. Aponta para o servidor onde o Registry está
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    model_name = "RecommendationSystemModel"
    stage = "Production"

    print(f"Conectando ao Registry para baixar '{model_name}' em '{stage}'...")

    # 2. A "URI Mágica": não usamos caminhos de pastas locais!
    model_uri = f"models:/{model_name}/{stage}"

    try:
        # 3. Baixa e carrega o modelo em memória
        loaded_model = mlflow.sklearn.load_model(model_uri)

        print("\nModelo carregado com sucesso a partir da Production.")
        print(f"Classe do modelo: {type(loaded_model).__name__}")
        print("\nO modelo esta pronto para receber o metodo .predict() em uma API!")

    except Exception as e:
        print(f"\nErro ao carregar o modelo: {e}")


if __name__ == "__main__":
    main()
