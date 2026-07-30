# Integração: MLflow Tracking e Model Registry 📊

Esta seção documenta a implementação da Issue #6. O ecossistema MLOps do projeto agora integra o **DVC** (para orquestração de pipeline e versionamento de dados) com o **MLflow** (para rastreamento de experimentos e registro de modelos), garantindo total reprodutibilidade e governança.

## 🏗️ 1. Arquitetura da Solução
Optamos por uma abordagem "DVC-First". O DVC continua sendo o maestro da execução via `uv run dvc repro`. O MLflow atua passivamente no final do estágio de avaliação (`evaluate`), observando as métricas geradas, capturando os artefatos físicos e gerenciando o ciclo de vida do modelo.

### Pré-requisitos
Certifique-se de que as dependências do projeto estão instaladas através do `uv`:
```bash
uv sync

```

---

## 🚀 2. Como Iniciar o Servidor MLflow

Para habilitar o *Model Registry* (Registro de Modelos), o MLflow requer um banco de dados relacional como backend. Utilizaremos o SQLite local.

Abra um terminal dedicado na raiz do projeto e execute:

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

```

* **Acesso à Interface:** Após iniciar o servidor, abra o navegador e acesse `http://localhost:5000`.
* **Nota de Segurança:** Os arquivos `mlflow.db` e o diretório `mlruns/` já foram adicionados ao `.gitignore` para evitar poluição no repositório.

---

## 🧪 3. Como Executar e Rastrear Experimentos

Não execute os scripts Python manualmente. Use o DVC para garantir que o pipeline e as dependências estejam sincronizados.

1. Altere os hiperparâmetros (ex: `n_neighbors` ou taxas de aprendizado) no arquivo `params.yaml`.
2. Execute o pipeline:
```bash
uv run dvc repro

```


3. O DVC re-executará apenas as etapas necessárias. No final, o script `scripts/dvc_evaluate.py` enviará automaticamente os parâmetros, as métricas e o modelo binário (usando formato `cloudpickle` para contornar restrições do `skops`) para o servidor MLflow.

### Comparando Experimentos

Para selecionar o melhor modelo, é necessário gerar múltiplas *runs*.

1. Altere o `params.yaml` e rode `uv run dvc repro` pelo menos **3 vezes** com valores diferentes.
2. Acesse a interface web (`http://localhost:5000`), selecione as execuções desejadas e clique em **Compare** para visualizar gráficos de dispersão e coordenadas paralelas.

---

## 🏆 4. Critério de Seleção e Governança de Modelos

### Critério Adotado

O critério matemático adotado para eleger o melhor modelo na transição para Produção é o maior **NDCG@10** (Normalized Discounted Cumulative Gain).

*Justificativa:* Como a avaliação utiliza a estratégia *Leave-One-Out* (apenas 1 item relevante escondido no conjunto de teste), a métrica de *Precision@10* é matematicamente esmagada. O NDCG@10 é a métrica mais robusta pois penaliza recomendações relevantes que aparecem no final da lista, garantindo que o modelo promovido é o que melhor ranqueia os itens no topo. O *Recall@10* é utilizado como métrica secundária de desempate.

### Promovendo o Modelo (Staging ➡️ Production)

Para automatizar a governança, criamos um script de gestão que atua como o "juiz". Ele varre o banco de dados do MLflow, encontra a *Run* com o maior NDCG@10, registra o modelo no *Model Registry* e promove sua versão para o estágio de Produção.

Execute o script de governança:

```bash
uv run python scripts/manage_registry.py

```

*Saída Esperada:* O terminal confirmará a melhor *run*, o nome do modelo vencedor e sua promoção para *Production*.

---

## 📥 5. Validando o Carregamento em Produção (Inferência)

Para garantir que o modelo está pronto para consumo por uma API (sem acoplamento a caminhos de arquivos locais), o projeto resgata o modelo dinamicamente do *Model Registry* utilizando a URI mágica estática (`models:/nome_do_modelo/estágio`).

Para validar a extração do modelo de Produção, execute:

```bash
uv run python scripts/predict.py

```

*Saída Esperada:* `Modelo carregado com sucesso a partir da Production.`

A partir deste momento, a variável do modelo instanciado no `predict.py` está pronta para receber dados reais através do método `.predict()`.
