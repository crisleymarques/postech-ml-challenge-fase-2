# Baselines e Estratégia de Avaliação

## Objetivo

Padronizar a comparação entre os modelos de recomendação do projeto, garantindo que os baselines e o futuro modelo neural sejam avaliados sob o mesmo protocolo, com os mesmos dados processados e o mesmo conjunto de teste.

## Baselines Implementados

### 1. Popularidade

- Estratégia: recomenda os itens com maior volume histórico de interações no conjunto de ajuste final.
- Vantagem: extremamente simples, reprodutível e um bom piso mínimo de desempenho.
- Limitação: não personaliza recomendações e tende a privilegiar itens já muito populares.

### 2. ItemKNN com Scikit-Learn

- Estratégia: utiliza `NearestNeighbors` do Scikit-Learn para construir similaridade item-item com cosseno a partir da matriz usuário-item.
- Geração de ranking: para cada usuário, soma as similaridades dos itens já consumidos com os demais candidatos.
- Vantagem: personaliza a recomendação e usa uma biblioteca padrão do ecossistema.
- Limitação: depende da coocorrência histórica e pode favorecer itens com maior conectividade no grafo de interações.

## Estratégia Única de Avaliação

- Seed global fixa: `42`.
- Dados de treino final para avaliação: `train + validation`.
- Dados de teste: `test`.
- Regra contra vazamento: o conjunto de teste é temporalmente posterior e os itens vistos em `train + validation` são removidos do ranking de recomendação no momento da avaliação.
- Todos os modelos devem implementar a mesma interface de `fit` e `recommend`.

## Métricas

As métricas foram escolhidas para cobrir qualidade do ranking e diversidade do catálogo:

### Precision@K

- Mede a proporção de itens relevantes dentro do top-K.
- Útil para estimar quão limpo está o topo da lista.
- Limitação: penaliza fortemente listas maiores mesmo quando há poucos itens relevantes por usuário.

### Recall@K

- Mede a fração dos itens relevantes recuperados no top-K.
- Útil para estimar capacidade de recuperação.
- Limitação: neste projeto, como o teste leave-last-out contém um item por usuário, `Recall@K` tende a coincidir com `HitRate@K`.

### HitRate@K

- Mede se ao menos um item relevante aparece no top-K.
- Muito usada em recomendação implícita e em cenários leave-last-out.
- Limitação: não distingue a posição do acerto dentro da lista.

### NDCG@K

- Mede qualidade do ranking considerando a posição do item relevante.
- Recompensa acertos mais próximos do topo.
- Limitação: depende do valor de `K` e pode ficar menos intuitiva para públicos não técnicos.

### MRR@K

- Mede o inverso da posição do primeiro item relevante.
- É útil quando a prioridade é colocar um item certo muito cedo na lista.
- Limitação: ignora ganhos adicionais depois do primeiro acerto.

### CatalogCoverage@K

- Mede a fração do catálogo que aparece nas listas recomendadas.
- Ajuda a evitar comparação baseada apenas em acurácia com excesso de concentração.
- Limitação: cobertura alta não garante relevância para o usuário final.

## Artefatos Gerados

Ao executar o script `scripts/run_baselines.py`, são gerados:

- `data/processed/movielens/evaluation/baseline_results.csv`
- `data/processed/movielens/evaluation/baseline_recommendations.json`
- `data/processed/movielens/evaluation/evaluation_protocol.json`

Esses arquivos permitem comparação direta e reprodutível entre baselines e o futuro modelo neural.
