import mlflow
from mlflow.tracking import MlflowClient


def main():
    # 1. Configurar a conexão com o banco de dados do MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    experiment_name = "MovieLens_Recommender_Experiment"
    registry_model_name = "RecommendationSystemModel"

    # 2. Buscar o experimento pelo nome
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Erro: Experimento '{experiment_name}' não encontrado.")
        return

    print("Buscando a melhor execução (Run)...")
    # 3. Buscar todas as runs e ordenar pelo maior NDCG@10
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.ndcg_at_k DESC"],
        max_results=1  # Pega apenas a primeira (a melhor)
    )

    if not runs:
        print("Nenhuma run encontrada. Rode o 'dvc repro' primeiro.")
        return

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_ndcg = best_run.data.metrics.get("ndcg_at_k", 0.0)
    best_model_name = best_run.data.params.get("model_name", "modelo_desconhecido")

    print(f"🏆 Melhor modelo encontrado: {best_model_name}")
    print(f"📈 NDCG@K: {best_ndcg:.5f}")
    print(f"🆔 Run ID: {best_run_id}")

    # 4. Registrar o modelo no Model Registry
    # O artifact_uri foi salvo como "modelo_nome-do-modelo" no dvc_evaluate.py
    model_uri = f"runs:/{best_run_id}/modelo_{best_model_name}"

    print(f"\nRegistrando o modelo '{registry_model_name}' no MLflow...")
    model_version = mlflow.register_model(model_uri=model_uri, name=registry_model_name)

    # 5. Promover para Staging e depois para Produção
    print(f"Movendo versão {model_version.version} para Staging...")
    client.transition_model_version_stage(
        name=registry_model_name,
        version=model_version.version,
        stage="Staging"
    )

    print(f"Aprovando e promovendo versão {model_version.version} para Production! 🚀")
    client.transition_model_version_stage(
        name=registry_model_name,
        version=model_version.version,
        stage="Production"
    )

    print("\n✅ Ciclo de vida atualizado com sucesso!")


if __name__ == "__main__":
    main()