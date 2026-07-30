# Ambiente Containerizado (Docker + DVC + MLflow)

## Pre-requisitos

- Docker Desktop com suporte a Docker Compose
- Pasta `storage/dvc-remote` presente (remote local do DVC)

## Estrutura de persistencia

O ambiente usa bind mounts para persistir:

- `./mlflow` em `/mlflow` (contém `mlflow.db` e `mlruns/`)
- `./data` em `/app/data`
- `./models` em `/app/models`
- `./reports` em `/app/reports`
- `./storage` em `/storage` (remote local do DVC)

## Subir o MLflow Server

```bash
docker compose up -d mlflow
```

A UI fica disponível em `http://localhost:5000`.

## Executar o pipeline completo via DVC (com tracking no MLflow)

```bash
docker compose run --rm trainer
```

O container executa:

- `dvc pull data/raw.dvc`
- `dvc repro`

## Governanca do Model Registry e validacao de carga

```bash
docker compose run --rm -e MLFLOW_TRACKING_URI=http://mlflow:5000 trainer python scripts/manage_registry.py
docker compose run --rm -e MLFLOW_TRACKING_URI=http://mlflow:5000 trainer python scripts/predict.py
```

## Execucao do zero

```bash
docker compose down -v
docker compose build --no-cache
docker compose up -d mlflow
docker compose run --rm trainer
```

