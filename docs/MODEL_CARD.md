# Model Card — NCF (PyTorch) — Sistema de Recomendação (MovieLens 1M)

## Visão geral

* **Nome do modelo:** `pytorch-ncf` (Neural Collaborative Filtering)
* **Tipo:** Ranqueamento de itens (Top-K) / Filtragem Colaborativa Baseada em Representações
* **Frameworks:** PyTorch, scikit-learn (baselines), MLflow (tracking e registry), DVC (versionamento de dados/pipeline)
* **Código-fonte do modelo:** `src/models/recommender.py` e `src/models/factory.py`
* **Rotina de treino:** `scripts/train_neural.py` e `src/utils_trainer.py`
* **Objetivo:** Comparar baselines e um modelo neural (NCF) sob o mesmo protocolo de split e métricas de ranking, estimando a probabilidade de interação entre um utilizador e um filme para gerar uma lista Top-K altamente relevante.

## Problema e definição do alvo

* **Entidade:** Par Utilizador-Item (`user_idx`, `item_idx`)
* **Alvo (label):** Interação (ex: clique, visualização, rating convertido em interação implícita).
* **Referência (Dataset/Dataloader):** `src/dataset.py` (Classe `RecommendationDataset`)

## Dados

### Fontes

* **Dataset:** MovieLens 1M (GroupLens Research).
* **Arquivos:** O dataset é carregado de arquivos em `data/raw/`:
* `ratings.dat`: Formato `UserID::MovieID::Rating::Timestamp` (arquivo principal de interações).
* `movies.dat`: Títulos e gêneros.
* `users.dat`: Demografia (idade, gênero, ocupação).


* **Documentação e licença:** [DOCUMENTACAO - DATASET](https://www.google.com/search?q=DOCUMENTACAO%2520-%2520DATASET)

### Features e prevenção de vazamento

* Para evitar vazamento temporal e viés de validação, os itens vistos em treino e validação são estritamente removidos do ranqueamento final durante a avaliação no conjunto de teste.

### Pré-processamento

O pré-processamento ocorre centralizado via DVC (etapa `preprocess` rodando `scripts/dvc_preprocess.py`):

* Limpeza de nulos e remoção de registos duplicados.
* Transformação dos IDs originais em índices numéricos contínuos começando em zero (`user_idx`, `item_idx`), requisito estrito das camadas de *Embedding* do PyTorch.

## Arquitetura e Modelos Comparados

O projeto avalia diferentes abordagens:

* **Baselines (Scikit-Learn):**
* *Popularidade*: Recomenda os itens mais consumidos (não personaliza).
* *ItemKNN*: Similaridade por cosseno baseada no histórico.


* **Modelo Neural (PyTorch):**
* *NCF (Neural Collaborative Filtering)*: *Embeddings* latentes de Utilizadores e Itens $\rightarrow$ Concatenação $\rightarrow$ Camadas Densas (MLP) $\rightarrow$ ReLU $\rightarrow$ Dropout $\rightarrow$ Logit final.



## Processo de treino

### Divisão treino/validação/teste (Protocolo de Split)

* **Estratégia:** Split temporal por utilizador (*leave-last-k-out*).
* **Validação:** 1 item por utilizador (`validation_k=1`).
* **Teste:** 1 item por utilizador (`test_k=1`).
* Apenas os itens que sobram formam o conjunto de treino.

### Otimização e hiperparâmetros padrão

* Parâmetros injetados via `configs/config.yaml` e `params.yaml` (ex: `embedding_dim`, `lr`, `batch_size`).
* Semente global fixada rigorosamente: `seed = 42`.
* O modelo ItemKNN teve seu parâmetro `n_neighbors` otimizado e registado ao longo dos experimentos.

### Early stopping

* Interrompe o treino quando a métrica monitorada (ex: loss ou NDCG de validação) deixa de melhorar.
* Restaura automaticamente os melhores pesos observados (checkpointing).

## Inferência e saída

* O modelo produz logits que são ordenados de forma decrescente.
* O resultado prático é um ranqueamento de tamanho `K` (Top-10 recomendações).

## Métricas de avaliação

As seguintes métricas são reportadas em `K=10` no holdout de teste (`scripts/dvc_evaluate.py`):

* Precision@10
* Recall@10
* HitRate@10
* NDCG@10 (Métrica principal para definição do melhor modelo)
* MRR@10
* CatalogCoverage@10

*Observação metodológica:* No protocolo com exatamente 1 item relevante por utilizador isolado no teste, **Recall@K e HitRate@K tendem a coincidir**.

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

* **Pipeline de Dados e Treino:** Totalmente versionado e orquestrado com DVC. Referência: [`dvc.yaml`](https://www.google.com/search?q=../dvc.yaml)
* **Tracking e Governança:** Todos os experimentos e transições de estágio (Staging $\rightarrow$ Production) estão registados e detalhados. Referência: [`MLFLOW_TRACKING_MODEL_REGISTRY.md`](https://www.google.com/search?q=MLFLOW_TRACKING_MODEL_REGISTRY.md)
* **Seeds e Ambiente:** Semente fixada (`42`) e dependências seladas no `pyproject.toml` com lock file.

## Como treinar

O pipeline inteiro está integrado pelo DVC. Para rodar a rotina de dados, treinos e avaliação:

```bash
# Executa de ponta a ponta (preprocess, features, baselines, treino neural e evaluation)
uv run dvc repro

```

Ou apenas o módulo neural isolado:

```bash
uv run python scripts/train_neural.py

```

## Artefatos gerados

No MLflow (`http://localhost:5000`) e DVC, a execução regista:

* Parâmetros do pipeline definidos em `params.yaml`
* Métricas globais (`evaluation_metrics.json`)
* O modelo PyTorch encapsulado em cloudpickle via `mlflow.pytorch.log_model`
* Registo direto no *Model Registry* automatizado para o estágio `Production`.