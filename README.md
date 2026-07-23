# Postech ML Challenge - Fase 2

Este projeto estabelece a base técnica e os padrões de engenharia para o desenvolvimento do sistema de recomendação da Fase 2.

## Estrutura do Projeto

A organização dos diretórios segue a estrutura:
- `src/`: Módulos de código-fonte (ex: configurações, pré-processamento, etc.).
- `tests/`: Testes unitários e integrados utilizando `pytest`.
- `data/`: Diretório destinado ao armazenamento de dados (gerenciado pelo DVC).
- `models/`: Diretório para os pesos e artefatos de modelos.
- `configs/`: Configurações adicionais de treinamento e modelos.
- `scripts/`: Scripts utilitários de suporte (ex: validação do ambiente).

---

## Gerenciamento de Dependências com `uv`

Este projeto utiliza o [uv](https://docs.astral.sh/uv/) para gerenciar dependências de forma ultra-rápida.

### Instalação

1. Certifique-se de que possui o `uv` instalado. Se não tiver, pode instalar via Homebrew:
   ```bash
   brew install uv
   ```

2. Sincronize as dependências e o ambiente virtual:
   ```bash
   uv sync
   ```
   *Nota: O `uv` criará automaticamente o ambiente virtual `.venv` e instalará a versão do Python correta configurada no `pyproject.toml`.*

3. Ative o ambiente virtual:
   ```bash
   source .venv/bin/activate
   ```

---

## Validação de Ambiente

Para garantir que todas as dependências obrigatórias (`torch`, `scikit-learn`, `mlflow`, `dvc`) e configurações foram carregadas corretamente, execute:
```bash
uv run scripts/validate_env.py
```

---

## Como Rodar o Pipeline

O fluxo de execução dos modelos consiste na preparação dos dados, avaliação de baselines e por fim o treinamento e avaliação do modelo neural.

1. **Processar o Dataset** (MovieLens):
   ```bash
   uv run scripts/process_movielens.py
   ```

2. **Rodar Modelos Baseline** (Popularity e ItemKNN):
   ```bash
   uv run scripts/run_baselines.py
   ```

3. **Treinar o Modelo Neural** (NCF):
   O treinamento utiliza MLflow local para rastrear as métricas, e salva o melhor checkpoint automaticamente.
   ```bash
   uv run scripts/train_neural.py
   ```

4. **Avaliar o Modelo Neural**:
   Carrega o melhor modelo treinado e avalia contra o conjunto de testes.
   ```bash
   uv run scripts/evaluate_neural.py
   ```

---

## Qualidade de Código & Git Hooks

### Ruff Linter & Formatter
O Ruff é configurado em `pyproject.toml` para validar o código de acordo com o padrão Google Style de Docstrings e outras boas práticas PEP8.

Para rodar manualmente a verificação e formatação:
```bash
uv run ruff check .
uv run ruff format .
```

### Pre-commit Hooks
Para habilitar a validação automática de lint e formatação a cada commit:
1. Instale os hooks na pasta `.git`:
   ```bash
   uv run pre-commit install
   ```
2. Caso queira rodar manualmente todos os hooks em todos os arquivos:
   ```bash
   uv run pre-commit run --all-files
   ```

---

## Convenção de Commits Semânticos (Conventional Commits)

Neste repositório, adotamos o padrão de commits semânticos para manter o histórico de alterações legível e organizado. A estrutura do commit deve seguir:

```
<tipo>(<escopo>): <descrição curta>
```

### Tipos de Commits Aceitos:
- `feat`: Implementação de novas funcionalidades (ex: `feat(preprocessing): adiciona MinMaxScalerStrategy`).
- `fix`: Correção de bugs ou problemas (ex: `fix(config): corrige leitura de arquivo .env`).
- `docs`: Modificações apenas na documentação (ex: `docs(readme): adiciona instruções de instalação`).
- `style`: Alterações que não afetam o significado do código (espaços em branco, formatação, etc.).
- `refactor`: Alterações de código que não corrigem bugs nem adicionam funcionalidades.
- `test`: Adição de testes em falta ou correção de testes existentes.
- `chore`: Atualizações de tarefas de build, gerenciamento de pacotes ou ferramentas auxiliares (ex: `chore(deps): atualiza versão do pytorch`).
