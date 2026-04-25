# Women's Health AI Diagnostic Support

Projeto acadêmico pronto para produção para detecção de risco em saúde da mulher usando o conjunto de dados Breast Cancer Wisconsin. O repositório inclui exploração automatizada dos dados, pré-processamento, treinamento de modelos, explicabilidade, backend com FastAPI, testes, suporte a Docker e um dashboard web.

## Objetivo

Construir um pipeline de machine learning de ponta a ponta que:

- carregue programaticamente um conjunto de dados público relacionado à saúde da mulher
- classifique casos de câncer de mama como `benign` ou `malignant`
- priorize o recall da classe maligna para triagem de risco em saúde
- exponha predições e metadados do modelo por meio de uma API FastAPI
- forneça explicabilidade com feature importance e SHAP

## Stack Tecnológica

- Python 3.11+
- FastAPI
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn
- SHAP
- Docker
- Dashboard em HTML / CSS / JavaScript

## Conjunto de Dados

- Fonte: `sklearn.datasets.load_breast_cancer`
- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Método de carregamento: totalmente programático, sem etapa manual de download
- Remapeamento da variável alvo: `1 = malignant`, `0 = benign`

## Estrutura do Projeto

```text
.
|-- artifacts/
|-- reports/
|   |-- eda/
|   `-- explainability/
|-- scripts/
|   |-- serve.py
|   `-- train.py
|-- src/
|   `-- womens_health_ai/
|       |-- api/
|       |   |-- static/
|       |   |-- main.py
|       |   `-- routes.py
|       |-- core/
|       |-- data/
|       |-- explainability/
|       |-- features/
|       |-- modeling/
|       |-- reporting/
|       |-- schemas/
|       |-- services/
|       `-- settings.py
|-- tests/
|-- Dockerfile
|-- pyproject.toml
`-- README.md
```

## Cobertura Funcional

### 1. Exploração de Dados

- estatísticas descritivas exportadas para `reports/eda/descriptive_statistics.csv`
- análise de valores ausentes exportada para `reports/eda/missing_value_summary.csv`
- gráfico de distribuição das variáveis
- gráfico de distribuição das classes
- matriz de correlação em CSV e mapa de calor
- boxplots de medidas clinicamente relevantes por diagnóstico

### 2. Pré-processamento de Dados

- imputação pela mediana para atributos numéricos
- imputação pelo valor mais frequente para atributos categóricos derivados
- one-hot encoding para atributos categóricos derivados
- padronização para variáveis numéricas
- implementação com `ColumnTransformer` + `Pipeline` do Scikit-learn

### 3. Modelagem

- Logistic Regression
- Decision Tree
- divisão treino/teste com estratificação
- busca leve de hiperparâmetros usando recall como critério de seleção

### 4. Avaliação

- accuracy
- recall
- F1-score
- classification report
- confusion matrix

Baseline validado atual no conjunto de teste hold-out:

- Logistic Regression: accuracy `0.9737`, recall `0.9524`, F1-score `0.9639`
- Decision Tree: accuracy `0.9123`, recall `0.8333`, F1-score `0.8750`
- Modelo campeão selecionado: `logistic_regression`

Justificativa no contexto de saúde:

- Recall é a métrica principal porque deixar de identificar um caso maligno pode atrasar a intervenção e aumentar o risco para a paciente.
- F1-score também é incluído porque sistemas de saúde ainda precisam controlar falsos positivos sem perder capacidade de detectar casos malignos.
- Accuracy é reportada, mas não é o principal critério de seleção, porque um valor alto pode ainda esconder falsos negativos perigosos.

### 5. Explicabilidade

- feature importance global para cada modelo treinado
- gráfico SHAP summary bar
- gráfico SHAP beeswarm
- gráfico SHAP waterfall para um caso individual

### 6. API

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `GET /docs`
- `GET /` dashboard

## Como Executar

### Ambiente local

```powershell
python -m pip install -e .
python scripts/train.py
python scripts/serve.py
```

A API ficará disponível em `http://127.0.0.1:8000`.

### Uvicorn direto

```powershell
$env:PYTHONPATH = "src"
uvicorn womens_health_ai.api.main:app --reload
```

### Docker

```powershell
docker build -t womens-health-ai .
docker run -p 8000:8000 womens-health-ai
```

## Exemplo de Requisição de Predição

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body (@{
  mean_radius = 17.99
  mean_texture = 10.38
  mean_perimeter = 122.8
  mean_area = 1001.0
  mean_smoothness = 0.1184
  mean_compactness = 0.2776
  mean_concavity = 0.3001
  mean_concave_points = 0.1471
  mean_symmetry = 0.2419
  mean_fractal_dimension = 0.07871
  radius_error = 1.095
  texture_error = 0.9053
  perimeter_error = 8.589
  area_error = 153.4
  smoothness_error = 0.006399
  compactness_error = 0.04904
  concavity_error = 0.05373
  concave_points_error = 0.01587
  symmetry_error = 0.03003
  fractal_dimension_error = 0.006193
  worst_radius = 25.38
  worst_texture = 17.33
  worst_perimeter = 184.6
  worst_area = 2019.0
  worst_smoothness = 0.1622
  worst_compactness = 0.6656
  worst_concavity = 0.7119
  worst_concave_points = 0.2654
  worst_symmetry = 0.4601
  worst_fractal_dimension = 0.1189
} | ConvertTo-Json)
```

## Saídas Geradas

O treinamento gera:

- `artifacts/model_bundle.joblib`
- `artifacts/model_info.json`
- `reports/eda/*.csv`
- `reports/eda/*.png`
- `reports/explainability/*.csv`
- `reports/explainability/*.png`

## Testes

```powershell
python -m pytest
```

## Observações

- A API oferece suporte à tomada de decisão e à triagem, mas não substitui diagnóstico clínico final.
- Valores ausentes são tratados pelo pipeline de pré-processamento e também são aceitos nos payloads de inferência.
- Atributos categóricos são derivados das medições clínicas brutas para atender aos requisitos de pré-processamento em produção, mantendo o dataset público totalmente carregado de forma programática.
