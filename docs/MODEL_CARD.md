# Model Card - Sistema de Recomendacao (MovieLens 1M)

## Visao geral

Este projeto implementa um sistema de recomendacao de filmes com foco em ranking Top-K, usando o dataset MovieLens 1M e protocolos de avaliacao reproduziveis.

## Objetivo e uso pretendido

- Objetivo tecnico: comparar baselines e um modelo neural (NCF) sob o mesmo protocolo de split e metricas de ranking.
- Uso pretendido: estudo/benchmark academico; nao recomendado para uso comercial sem revisao de privacidade/viesses.

## Dados

- Dataset: MovieLens 1M (GroupLens Research).
- Arquivo principal de interacoes: `ratings.dat` no formato `UserID::MovieID::Rating::Timestamp`.
- Caracteristicas adicionais: `movies.dat` (titulo e generos) e `users.dat` (demografia).
- Documentacao e licenca: [DOCUMENTACAO - DATASET](DOCUMENTACAO%20-%20DATASET)

## Protocolo de split

- Split temporal por usuario (leave-last-k-out).
- Validacao: 1 item por usuario (`validation_k=1`)
- Teste: 1 item por usuario (`test_k=1`)
- Itens vistos em treino/validacao sao removidos do ranking durante a avaliacao.
- Seed global fixa: 42.

## Modelos

- Baselines:
  - Popularidade (nao personaliza)
  - ItemKNN (Scikit-Learn, similaridade cosseno)
- Modelo neural:
  - NCF (embeddings de usuario/item + MLP)

## Metricas

Metricas reportadas em `K=10`:

- Precision@K
- Recall@K
- HitRate@K
- NDCG@K
- MRR@K
- CatalogCoverage@K

Observacao: no protocolo com 1 item relevante por usuario no teste, Recall@K e HitRate@K tendem a coincidir.

## Resultados (referencia)

Resultados mais recentes gerados pelo pipeline (ver `reports/metrics/evaluation_metrics.json`):

- Popularidade: NDCG@10 = 0.01740
- ItemKNN: NDCG@10 = 0.03330

Os resultados variam conforme hiperparametros (ex.: `n_neighbors`) e configuracoes de treino.

## Limitacoes e riscos

- Dataset nao reflete catalogos de e-commerce e nao contem contexto de sessao.
- Demografia em `users.dat` e limitada e auto-reportada.
- Baselines podem favorecer itens populares e reduzir diversidade.
- NCF pode amplificar vieses de popularidade presentes no historico.

## Reprodutibilidade

- Pipeline com DVC: [dvc.yaml](../dvc.yaml)
- Tracking/Registry com MLflow: [MLFLOW_TRACKING_MODEL_REGISTRY.md](MLFLOW_TRACKING_MODEL_REGISTRY.md)
