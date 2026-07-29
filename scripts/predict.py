import mlflow


def main():
    # 1. Aponta para o servidor onde o Registry está
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    model_name = "RecommendationSystemModel"
    stage = "Production"

    print(f"Conectando ao Registry para baixar '{model_name}' em '{stage}'...")

    # 2. A "URI Mágica": não usamos caminhos de pastas locais!
    model_uri = f"models:/{model_name}/{stage}"

    try:
        # 3. Baixa e carrega o modelo em memória
        loaded_model = mlflow.sklearn.load_model(model_uri)

        print("\n✅ Modelo carregado com sucesso a partir da Produção!")
        print(f"🤖 Classe do Modelo: {type(loaded_model).__name__}")
        print("\nO modelo está pronto para receber o método .predict() em uma API!")

    except Exception as e:
        print(f"\n❌ Erro ao carregar o modelo: {e}")


if __name__ == "__main__":
    main()