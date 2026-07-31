# Postech ML Challenge - Fase 2 (Sistema de Recomendacao)

Este repositório implementa um sistema de recomendacao com foco em ranking Top-K, usando o dataset MovieLens 1M e boas praticas de reprodutibilidade (DVC, MLflow e Docker).

## Problema e objetivos

- Problema de negocio (simulado): recomendar itens (filmes) para usuarios com base no historico de interacoes.
- Objetivos tecnicos:
  - construir um pipeline reproduzivel de dados -> treino -> avaliacao
  - padronizar baselines e metricas para comparacao com um modelo neural (NCF)
  - rastrear experimentos e governar o melhor modelo via MLflow Tracking e Model Registry
  - disponibilizar um ambiente containerizado para executar tudo do zero

## Dataset

- Dataset: MovieLens 1M (GroupLens Research).
- Volume: 1,000,209 avaliacoes, 6,040 usuarios e ~3,900 filmes (MovieLens 1M).
- Origem/licenca/estrutura: [DOCUMENTACAO - DATASET](docs/DOCUMENTACAO%20-%20DATASET)
- Formatos:
  - `ratings.dat`: `UserID::MovieID::Rating::Timestamp`
  - `users.dat`: demografia (auto-reportada)
  - `movies.dat`: titulo e generos

## Arquitetura da solucao

- Pipeline de dados: leitura, limpeza, features e split temporal leave-last-k-out por usuario em [movielens_pipeline.py](src/data/movielens_pipeline.py)
- Modelos:
  - baselines em [baselines.py](src/models/baselines.py)
  - modelo neural NCF em [ncf.py](src/models/ncf.py) e estrategia de treino em [strategy.py](src/training/strategy.py)
- Avaliacao unica (Top-K): metricas e rotina comum em [recommender_evaluation.py](src/evaluation/recommender_evaluation.py)
- Orquestracao/reprodutibilidade: DVC executa as etapas via [dvc.yaml](dvc.yaml)
- Tracking e Registry: MLflow registra runs, metricas e modelos no stage `evaluate` e governa o melhor modelo via [MLFLOW_TRACKING_MODEL_REGISTRY.md](docs/MLFLOW_TRACKING_MODEL_REGISTRY.md)

## Estrutura do repositorio

- `src/`: codigo do pipeline, modelos, avaliacao e configuracao
- `scripts/`: scripts executaveis (DVC stages, treino neural, governanca do registry)
- `data/`: dados (rastreados por DVC)
- `models/`: modelos serializados e checkpoints
- `reports/`: relatorios e metricas consolidadas
- `docs/`: documentacao do projeto
- `tests/`: testes automatizados (pytest)
- `notebooks/`: EDA e exploracao

## Pre-requisitos

- Python >= 3.10 (recomendado 3.11)
- Git
- `uv` (recomendado) ou Poetry (opcional)
- Docker Desktop + Docker Compose (opcional, para execucao containerizada)

## Instalacao

### Opcao A (recomendada): uv

```bash
pip install uv
uv sync
```

### Opcao B (opcional): Poetry

Este projeto usa `pyproject.toml` no formato PEP 621. Se voce preferir Poetry, use uma versao com suporte a PEP 621. Caso encontre incompatibilidades, use a instalacao via `uv`.

## Configuracao de variaveis de ambiente (.env)

1. Copie o template:

```bash
copy .env.example .env
```

2. Ajuste conforme necessario (valores comuns):

- `MLFLOW_TRACKING_URI`
  - local (sem servidor): `sqlite:///mlflow.db`
  - com servidor (compose): `http://localhost:5000`
- `MLFLOW_EXPERIMENT_NAME`: nome do experimento (padrao do projeto)
- `DVC_REMOTE_PATH`: caminho do remote (local) ou url S3

O carregamento do `.env` e centralizacao de configuracoes ficam em [config.py](src/config.py).

## Validacao do ambiente

Valida diretorios, imports e variaveis carregadas:

```bash
uv run python scripts/validate_env.py
```

## Qualidade de codigo (lint, format, pre-commit) e testes

Lint e format:

```bash
uv run ruff check .
uv run ruff format .
```

Pre-commit:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

Testes:

```bash
uv run pytest -q
```

## DVC: dados, remote e execucao do pipeline

### Obter dados

Baixar dados do remote configurado:

```bash
uv run dvc pull data/raw.dvc
```

O remote padrao do projeto e local (`storage/dvc-remote`). Se voce estiver em outra maquina, voce precisa copiar/sincronizar essa pasta manualmente.

### Configurar remote (local ou S3)

Listar remotes:

```bash
uv run dvc remote list
```

Trocar remote local:

```bash
uv run dvc remote remove localstorage
uv run dvc remote add -d localstorage storage/dvc-remote
```

S3 (credenciais sempre em modo `--local`, nunca em arquivos versionados):

```bash
uv run dvc remote modify localstorage url s3://meu-bucket/dados
uv run dvc remote modify --local localstorage access_key_id <AWS_ACCESS_KEY_ID>
uv run dvc remote modify --local localstorage secret_access_key <AWS_SECRET_ACCESS_KEY>
```

### Executar pipeline completo

```bash
uv run dvc repro
```

Saidas principais:

- `data/interim/movielens/`
- `data/processed/movielens/`
- `models/baselines/`
- `reports/evaluation/movielens/`
- `reports/metrics/*.json`

Detalhes do pipeline: [DVC_PIPELINE.md](docs/DVC_PIPELINE.md)

## MLflow: Tracking, UI e Model Registry

### Subir servidor MLflow (habilita Registry)

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

Abrir a UI: `http://localhost:5000`.

### Gerar runs (minimo 3)

1. Altere hiperparametros em `params.yaml` (ex.: `train.item_knn.n_neighbors`)
2. Execute:

```bash
uv run dvc repro
```

### Selecionar melhor run e promover para Production

```bash
uv run python scripts/manage_registry.py
```

### Recuperar modelo em Production (inferenca)

```bash
uv run python scripts/predict.py
```

Guia completo: [MLFLOW_TRACKING_MODEL_REGISTRY.md](docs/MLFLOW_TRACKING_MODEL_REGISTRY.md)

## Docker Compose (ambiente reproduzivel do zero)

Guia completo: [DOCKER_DVC_MLFLOW.md](docs/DOCKER_DVC_MLFLOW.md)

Comandos principais:

```bash
docker compose build
docker compose up -d mlflow
docker compose run --rm trainer
```

UI do MLflow: `http://localhost:5000`.

## Resultados e comparacao com baselines

Os resultados consolidados do pipeline ficam em `reports/metrics/evaluation_metrics.json`.

Referencia (K=10, seed=42, parametros atuais do repo):

| modelo | precision@10 | recall@10 | hitrate@10 | ndcg@10 | mrr@10 | catalog_coverage@10 |
| --- | --- | --- | --- | --- | --- | --- |
| popularity | 0.00353 | 0.03526 | 0.03526 | 0.01740 | 0.01213 | 0.05343 |
| item_knn_sklearn | 0.00626 | 0.06258 | 0.06258 | 0.03330 | 0.02439 | 0.14031 |

Detalhes das escolhas de metricas e limitacoes: [BASELINES_E_AVALIACAO.md](docs/BASELINES_E_AVALIACAO.md)

## Model Card

- [MODEL_CARD.md](docs/MODEL_CARD.md)

## Video final

- Link: (preencher)

## Deploy (opcional)

- URL: (preencher)
- Instrucoes: (preencher)
