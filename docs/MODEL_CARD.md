# Model Card — NCF (PyTorch) — Sistema de Recomendação (MovieLens 1M)

## Visão geral
Este projeto implementa um sistema de recomendacao de filmes com foco em ranking Top-K, usando o dataset MovieLens 1M e protocolos de avaliacao reproduziveis.
* **Nome do modelo:** `pytorch-ncf` (Neural Collaborative Filtering)
* **Tipo:** Ranqueamento de itens (Top-K) / Filtragem Colaborativa Baseada em Representações
* **Frameworks:** PyTorch, scikit-learn (baselines), MLflow (tracking e registry), DVC (versionamento de dados/pipeline)
* **Código-fonte do modelo:** [ncf.py](../src/models/ncf.py) (arquitetura neural) e [baselines.py](../src/models/baselines.py)
* **Rotina de treino (loop + early stopping):** [strategy.py](../src/training/strategy.py)
* **Objetivo:** Comparar baselines e um modelo neural (NCF) sob o mesmo protocolo de split e métricas de ranking, estimando a probabilidade de interação entre um usuario e um filme para gerar uma lista Top-K altamente relevante.

## Problema e definição do alvo

* **Entidade:** Par usuario-item (`user_idx`, `item_idx`)
* **Alvo (label):** Interacao implicita (coluna `interaction`, tipicamente 0/1) gerada no processamento.
* **Referência (Dataset/Dataloader):** [recsys_dataset.py](../src/data/recsys_dataset.py) (classe `InteractionDataset`)

## Dados

### Fontes

* **Dataset:** MovieLens 1M (GroupLens Research).
* **Arquivos:** O dataset é carregado de arquivos em `data/raw/`:
* `ratings.dat`: Formato `UserID::MovieID::Rating::Timestamp` (arquivo principal de interações).
* `movies.dat`: Títulos e gêneros.
* `users.dat`: Demografia (idade, gênero, ocupação).


### Features e prevenção de vazamento

* Para evitar vazamento temporal e vies de validacao, os itens vistos em treino e validacao sao estritamente removidos do ranqueamento final durante a avaliacao no conjunto de teste.

### Pré-processamento

O pré-processamento ocorre centralizado via DVC:

* Limpeza de nulos e remoção de registos duplicados.
* Transformação dos IDs originais em índices numéricos contínuos começando em zero (`user_idx`, `item_idx`), requisito estrito das camadas de *Embedding* do PyTorch.

## Arquitetura e Modelos Comparados

O projeto avalia diferentes abordagens (definidas em `src/models/`):

* **Baselines (Scikit-Learn):**
* *Popularidade*: Recomenda os itens mais consumidos (não personaliza).
* *ItemKNN*: Similaridade por cosseno baseada no histórico.


* **Modelo Neural (PyTorch):**
* *NCF (Neural Collaborative Filtering)*: embeddings de usuario e item + camadas densas (MLP) para produzir um score final.



## Processo de treino

### Divisão treino/validação/teste (Protocolo de Split)

* **Estrategia:** Split temporal por usuario (leave-last-k-out).
* **Validacao:** 1 item por usuario (`validation_k=1`).
* **Teste:** 1 item por usuario (`test_k=1`).
* Apenas os itens que sobram formam o conjunto de treino.

### Otimização e hiperparâmetros padrão

* Baselines e pipeline DVC: hiperparametros em `params.yaml` (ex.: `train.item_knn.n_neighbors`).
* NCF (PyTorch): hiperparametros via `Settings` e `.env` (ver [config.py](../src/config.py) e `.env.example` na raiz do projeto), ex.: `learning_rate`, `batch_size`, `embedding_dim`, `epochs`, `early_stopping_patience`.
* O modelo ItemKNN teve seu parâmetro `n_neighbors` otimizado e registado ao longo dos experimentos.

### Early stopping

* Implementado e orquestrado dentro da estrategia de treino em `src/training/strategy.py`.
* Interrompe o treino quando a métrica monitorada (ex: loss ou NDCG de validação) deixa de melhorar.
* Restaura automaticamente os melhores pesos observados.

## Inferência e saída

* O modelo produz logits que são ordenados de forma decrescente.
* O resultado prático é um ranqueamento de tamanho `K` (Top-10 recomendações).

## Métricas de avaliação

As métricas são reportadas em `K=10` no holdout de teste através do módulo `src/evaluation/recommender_evaluation.py`:

* Precision@10
* Recall@10
* HitRate@10
* NDCG@10 (Métrica principal para definição do melhor modelo)
* MRR@10
* CatalogCoverage@10

*Observacao metodologica:* No protocolo com exatamente 1 item relevante por usuario isolado no teste, **Recall@K e HitRate@K tendem a coincidir**.

### Resultados (referência a partir do MLflow)

Resultados mais recentes gerados pelo pipeline (ver `reports/metrics/evaluation_metrics.json`):

* **Popularidade:** NDCG@10 = `0.01740`
* **ItemKNN:** NDCG@10 = `0.03330`
* *(Os resultados do modelo NCF variam conforme os hiperparâmetros e configurações de treino gravadas no MLflow).*

## Uso pretendido

* **Uso adequado:** Estudo e benchmark acadêmico/técnico focado na comparação de baselines clássicas contra abordagens baseadas em redes neurais de representação latente.
* **Uso não recomendado:** Uso comercial direto em produção sem revisão de privacidade, viéses e regras de negócios adicionais (filtros de catálogo, disponibilidade de estoque).

## Considerações de segurança, privacidade e ética

* A demografia presente no `users.dat` é limitada e auto-reportada. É fundamental ter cuidado para que o modelo não crie agrupamentos enviesados com base em atributos protegidos (ex: gênero) caso as features demográficas sejam incorporadas à rede no futuro.
* **Risco de Viés de Popularidade:** O uso prolongado de modelos de filtragem colaborativa (como NCF e ItemKNN) pode amplificar o *feedback loop* histórico, esmagando a diversidade e soterrando a cauda longa de itens menos conhecidos.

## Limitações e riscos conhecidos

* **Contexto:** O dataset utilizado (MovieLens) é voltado para avaliações de filmes, logo, não reflete a dinâmica transacional exata de catálogos de e-commerce (como abandono de carrinho ou recompra) e não contém contexto de sessão contínua.
* **Cold Start:** Não consegue recomendar itens recém-adicionados ou atender novos clientes sem depender do baseline estático de Popularidade.

## Reprodutibilidade

* **Pipeline DVC (dados + baselines):** versionado e orquestrado com DVC. Referência: [dvc.yaml](../dvc.yaml)
* **Sementes e Aleatoriedade:** semente fixada (`seed = 42`) e utilitarios em [seeds.py](../src/utils/seeds.py).
* **Tracking e Governança (MLflow):** experimentos e transicoes (Staging -> Production) documentados em [MLFLOW_TRACKING_MODEL_REGISTRY.md](MLFLOW_TRACKING_MODEL_REGISTRY.md).
* **Ambiente:** Dependências seladas usando Pydantic/uv no `pyproject.toml` e lock file.

## Como treinar

O pipeline DVC cobre dados, features, treino e avaliacao dos baselines. Para executar:

```bash
# Executa preprocess -> feature_eng -> train -> evaluate (baselines)
uv run dvc repro

```

Para treinar o NCF (fora do pipeline DVC atual), rode o script neural:

```bash
uv run python scripts/train_neural.py

```

Para avaliar o NCF sob o mesmo protocolo de recomendadores:

```bash
uv run python scripts/evaluate_neural.py
```

## Artefatos gerados

No MLflow (`http://localhost:5000`) e DVC, a execução regista:

* Parâmetros do pipeline definidos em `params.yaml`
* Métricas globais (`evaluation_metrics.json`)
* Baselines: modelos logados no MLflow no stage `evaluate` do DVC (ver `scripts/dvc_evaluate.py`) e promovidos via `scripts/manage_registry.py` para `Production` quando aplicavel.
* NCF: o checkpoint `models/ncf_best.pth` e registrado como artifact no run `ncf_training` (ver `scripts/train_neural.py`).
