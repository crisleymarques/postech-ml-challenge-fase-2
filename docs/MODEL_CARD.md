# Model Card — NCF (PyTorch) — Sistema de Recomendação

## Visão geral

* **Nome do modelo:** `pytorch-ncf` (Neural Collaborative Filtering)
* **Tipo:** Ranqueamento de itens (Top-K) / Filtragem Colaborativa Baseada em Representações
* **Frameworks:** PyTorch, scikit-learn (baselines), MLflow (tracking e registry), DVC (versionamento de dados/pipeline)
* **Código-fonte do modelo:** `src/models/recommender.py` e `src/models/factory.py`
* **Rotina de treino (loop + early stopping):** `scripts/train_neural.py` e `src/utils_trainer.py`
* **Objetivo:** Estimar a probabilidade de interação entre um utilizador e um item para gerar uma lista ranqueada (Top-K) altamente relevante, aumentando o engajamento e a descoberta do catálogo.

## Problema e definição do alvo

* **Entidade:** Par Utilizador-Item (`user_idx`, `item_idx`)
* **Alvo (label):** Interação implícita/explícita (ex: clique, visualização, rating convertido em 1). Para validação, utiliza-se a avaliação contra amostras negativas (itens com os quais não interagiu).
* **Referência (Dataset/Dataloader):** `src/dataset.py` (Classe `RecommendationDataset`)

## Dados

### Fontes

* Os dados originais (ex: MovieLens 1M ou similar de e-commerce) são lidos da pasta `data/raw/` e processados pelo pipeline do DVC.
* O versionamento ponta a ponta dos dados é controlado via arquivo `dvc.yaml`.

### Features e prevenção de vazamento

* Para evitar vazamento temporal e viés de validação, a separação de treino/teste utiliza o protocolo **Leave-One-Out** ordenado pelo tempo (`timestamp`). O último item consumido pelo utilizador é isolado para teste.
* O mapeamento dos IDs cria dicionários rigorosos mantidos em memória e persistidos como artefatos para inferência.

### Pré-processamento

O pré-processamento ocorre centralizado via DVC (etapa `preprocess` rodando `scripts/dvc_preprocess.py`):

* Limpeza de nulos e remoção de registos duplicados.
* Transformação dos IDs alfanuméricos (`user_id`, `item_id`) em índices numéricos contínuos começando em zero (`user_idx`, `item_idx`), requisito estrito das camadas de *Embedding* do PyTorch.

## Arquitetura do modelo

Rede Neural de Filtragem Colaborativa (NCF) simples:

* **Entrada:** Dois tensores categóricos (`user_idx` e `item_idx`).
* **Camadas:**
* *Embeddings* latentes para Utilizadores e Itens.
* Concatenação dos embeddings $\rightarrow$ Camadas Densas (MLP - *Multi-Layer Perceptron*) $\rightarrow$ ReLU $\rightarrow$ Dropout.


* **Saída:** **Logit** (Score de similaridade/probabilidade de interação).
* **Referência:** Classe `NCFRecommender(nn.Module)` estruturada dinamicamente pelo padrão de projeto em `ModelFactory`.

## Processo de treino

### Divisão treino/validação/teste

* Estratégia **Leave-One-Out** (último item para teste, penúltimo para validação, o restante para treino).
* Os itens de teste são avaliados juntamente com amostras negativas isoladas durante a divisão.

### Otimização e hiperparâmetros padrão

Geridos externalmente e injetados na rotina de treino via `configs/config.yaml` e `params.yaml`:

* `embedding_dim` (ex: 64 ou 128)
* `lr = 0.001`
* `batch_size = 256` (ou ajustado conforme memória)
* `epochs = 50` (máximo)
* Semente global fixada (`seed: 42`)
* **Referência:** `configs/config.yaml`

### Early stopping

* Implementado via classe auxiliar `EarlyStopping` em `src/utils_trainer.py`.
* Interrompe o treino quando a *loss* de validação (ou NDCG de validação) deixa de melhorar por *X* épocas (`patience`).
* Restaura automaticamente os melhores pesos observados (checkpointing).

## Inferência e saída

* O modelo recebe um utilizador e um catálogo de itens candidatos, produzindo logits que são ordenados de forma decrescente.
* O resultado prático é um ranqueamento de tamanho `K` (Top-10 recomendações).
* **Referência:** Script `scripts/predict.py`.

## Métricas de avaliação

O script `scripts/dvc_evaluate.py` reporta e regista as seguintes métricas no **conjunto de teste**:

* NDCG@10 (Métrica principal e critério de seleção para Produção)
* Hit Rate@10 / Recall@10
* Precision@10 (Naturalmente subestimado pelo Leave-One-Out)
* Catalog Coverage (Cobertura de Catálogo)

### Resultados (a partir do MLflow)

Os valores ficam registados no MLflow, comparando o modelo Neural com as baselines (`popularity` e `item_knn_sklearn`).

* **MLflow Tracking URI:** `http://localhost:5000` (Servidor local SQLite configurado na raiz)
* **Desempenho Baselines (Exemplo):**
* Popularidade: NDCG@10 ~0.017 | Catalog Coverage ~5.3%
* KNN: NDCG@10 ~0.033 | Catalog Coverage ~15.19%


* **Critério de Seleção:** O script `scripts/manage_registry.py` analisa automaticamente as *runs*, identifica a de **maior NDCG@10**, regista no Registry (como `RecommendationSystemModel`), promove para `Staging` e depois para `Production`.

## Uso pretendido

* **Uso adequado:** Prateleiras personalizadas (ex: "Recomendados para si") para utilizadores autenticados com um histórico mínimo; E-mail marketing segmentado.
* **Uso não recomendado:** Utilizadores anónimos ou recém-registados (*User Cold Start*) sem um modelo de *fallback* (como itens populares ou KNN por região); Substituição direta de busca por termo explícito.

## Considerações de segurança, privacidade e ética

* Vieses (*Feedback Loop*): Existe risco de viés de popularidade a longo prazo. Se o modelo aprender a recomendar apenas blockbusters e os utilizadores clicarem neles, a "cauda longa" do catálogo será invisibilizada nos próximos treinos.
* Privacidade: O uso de Embeddings mitiga a necessidade de passar atributos diretos de identificação demográfica, usando apenas os índices anónimos, porém deve-se evitar cruzar esses IDs numéricos com tabelas de PII fora do ambiente seguro.

## Limitações e riscos conhecidos

* **Cold Start:** Não consegue recomendar itens recém-adicionados (*Item Cold Start*) nem prever de imediato para novos clientes.
* **Sparsity:** O desempenho degrada-se visivelmente para itens de nicho com pouquíssimas interações anteriores.
* **Custos Computacionais:** Retreinar frequentemente uma arquitetura neural exige infraestrutura com suporte a GPU, sendo mais custoso do que atualizar vizinhanças de baselines KNN.

## Reprodutibilidade

* Seeds fixadas de forma centralizada: `src/utils.py` com a rotina `set_seed(seed: 42)` que cobre Python, NumPy e PyTorch.
* Ambiente selado com `pyproject.toml` (Poetry/uv) e `lock file`.
* Toda a *pipeline* de dados, modelo e métricas selada com o **DVC** no arquivo `dvc.yaml`.

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