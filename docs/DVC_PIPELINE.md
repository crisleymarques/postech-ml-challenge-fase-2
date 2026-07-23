# Pipeline DVC

## Objetivo

Documentar como o projeto versiona dados e reproduz as etapas de processamento, treinamento e avaliação com DVC.

## Estrutura do Pipeline

- `preprocess`: lê `data/raw`, executa profiling e limpeza e gera artefatos em `data/interim/movielens/`
- `feature_eng`: gera features e splits temporais e publica artefatos em `data/processed/movielens/`
- `train`: treina os modelos baseline e salva os binários em `models/baselines/`
- `evaluate`: aplica o mesmo protocolo de avaliação aos modelos treinados e grava relatórios em `reports/evaluation/movielens/`

Os parâmetros centralizados ficam em `params.yaml`.

## Remote

O remote padrão do repositório é local:

```bash
python -m dvc remote list
```

Configuração atual:

- nome: `localstorage`
- url: `storage/dvc-remote`

Para trocar por outro diretório local:

```bash
python -m dvc remote remove localstorage
python -m dvc remote add -d localstorage storage/dvc-remote
```

Para usar S3:

```bash
python -m dvc remote modify localstorage url s3://meu-bucket/dados
python -m dvc remote modify --local localstorage access_key_id <AWS_ACCESS_KEY_ID>
python -m dvc remote modify --local localstorage secret_access_key <AWS_SECRET_ACCESS_KEY>
```

## Fluxo de Execução

Sincronizar dados e artefatos a partir do remote:

```bash
python -m dvc pull
```

Executar o pipeline completo:

```bash
python -m dvc repro
```

Executar um stage específico:

```bash
python -m dvc repro train
```

Enviar atualizações para o remote:

```bash
python -m dvc push
```

## Saídas por Stage

- `preprocess`
  - output: `data/interim/movielens/`
  - métricas: `reports/metrics/preprocess_metrics.json`
- `feature_eng`
  - output: `data/processed/movielens/`
  - métricas: `reports/metrics/feature_eng_metrics.json`
- `train`
  - output: `models/baselines/`
  - métricas: `reports/metrics/train_metrics.json`
- `evaluate`
  - output: `reports/evaluation/movielens/`
  - métricas: `reports/metrics/evaluation_metrics.json`

## Artefatos Versionados

- dataset bruto: `data/raw.dvc`
- lock do pipeline: `dvc.lock`
- definição do pipeline: `dvc.yaml`

## Observações

- o split usado no pipeline é temporal `leave-last-k-out`, com `validation_k=1` e `test_k=1`
- a seed global do projeto permanece fixa em `42`
- os artefatos grandes são ignorados pelo Git e rastreados pelo DVC
