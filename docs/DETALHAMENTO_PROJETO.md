# Detalhamento Técnico do Projeto

## 1. Visão Geral

Este projeto foi construído como um sistema de apoio à decisão para saúde da mulher, com foco em classificação de risco relacionada a câncer de mama. A solução combina:

- pipeline de machine learning para dados tabulares
- explicabilidade com SHAP e feature importance
- API de inferência com FastAPI
- dashboard web para uso interativo
- artefatos reprodutíveis de EDA, treinamento e avaliação

O objetivo não é substituir diagnóstico médico, e sim oferecer uma base técnica sólida para triagem, apoio analítico e demonstração de boas práticas de engenharia de software e machine learning.

## 2. Problema Atacado

O problema tratado é de classificação binária: identificar se um conjunto de medidas clínicas está mais associado a um caso `benign` ou `malignant`.

Em contexto de saúde, o custo de erro não é simétrico:

- um falso positivo gera investigação adicional
- um falso negativo pode atrasar diagnóstico e tratamento

Por isso, o projeto foi estruturado para priorizar `recall` da classe maligna como principal critério de seleção do modelo campeão.

## 3. Por Que Esse Dataset Foi Escolhido

Foi utilizado o dataset `Breast Cancer Wisconsin (Diagnostic)`, carregado programaticamente via `sklearn.datasets.load_breast_cancer`.

Essa escolha foi feita porque:

- é um dataset público, confiável e amplamente usado em ensino e pesquisa
- está diretamente relacionado à saúde da mulher
- possui medidas clínicas reais de massa mamária
- é adequado para classificação supervisionada
- permite construir um pipeline completo sem etapas manuais de aquisição de dados

Além disso, ele é muito apropriado para um Tech Challenge porque permite demonstrar:

- exploração de dados
- pré-processamento
- treinamento de múltiplos modelos
- avaliação com métricas clínicas relevantes
- explicabilidade
- exposição em API

## 4. Arquitetura da Solução

O projeto foi dividido em camadas com responsabilidades claras para evitar acoplamento desnecessário e facilitar manutenção.

### 4.1 Estrutura principal

```text
src/womens_health_ai/
|-- api/
|-- core/
|-- data/
|-- explainability/
|-- features/
|-- modeling/
|-- reporting/
|-- schemas/
|-- services/
|-- utils/
`-- settings.py
```

### 4.2 Motivo da estrutura

Essa organização foi escolhida para separar preocupações:

- `data/`: carregamento do dataset
- `features/`: engenharia de atributos
- `modeling/`: treino, seleção, avaliação e persistência de modelos
- `reporting/`: geração de análises exploratórias e relatórios
- `explainability/`: feature importance e SHAP
- `schemas/`: contratos de entrada e saída da API
- `services/`: lógica de inferência
- `api/`: rotas e assets do frontend
- `utils/`: funções utilitárias genéricas
- `settings.py`: parâmetros globais e resolução de caminhos

Essa separação foi feita para que:

- o treinamento não fique misturado com a camada HTTP
- a inferência possa ser reaproveitada fora da API
- a validação de payloads fique centralizada
- a evolução futura do projeto seja previsível

## 5. Fluxo de Dados do Projeto

O fluxo principal foi implementado desta forma:

1. carregar o dataset público
2. normalizar nomes de colunas
3. remapear a variável alvo para `1 = malignant` e `0 = benign`
4. gerar atributos derivados
5. dividir os dados em treino e teste com estratificação
6. construir pipelines de pré-processamento e modelagem
7. treinar dois modelos candidatos
8. avaliar com métricas clínicas e gerais
9. selecionar o modelo campeão com base em `recall`
10. gerar artefatos de explicabilidade e EDA
11. persistir os modelos e metadados
12. servir inferência via FastAPI

Esse fluxo foi escolhido porque garante rastreabilidade entre dados, modelo, métricas e API.

## 6. Decisões de Pré-processamento

## 6.1 Tratamento de valores ausentes

Mesmo que o dataset utilizado não apresente um volume relevante de valores ausentes, o pipeline foi preparado para ambiente mais próximo de produção.

As escolhas foram:

- imputação por mediana para atributos numéricos
- imputação pelo valor mais frequente para atributos categóricos derivados

Isso foi feito porque:

- APIs reais podem receber payloads incompletos
- a robustez de inferência não deve depender da perfeição do dado de entrada
- mediana é mais robusta a outliers do que média

## 6.2 Escalonamento

Foi aplicado `StandardScaler` nas variáveis numéricas.

Motivo:

- Logistic Regression é sensível à escala das features
- o escalonamento melhora estabilidade numérica
- manter um pipeline padronizado ajuda consistência entre treino e inferência

## 6.3 Codificação categórica

O dataset original é totalmente numérico. No entanto, o desafio exigia codificação de variáveis categóricas.

Para atender esse requisito de forma tecnicamente coerente, foram criadas features categóricas derivadas:

- `radius_band`
- `texture_band`
- `concavity_band`

Essas variáveis foram construídas a partir de faixas clínicas simples usando `pd.cut`, e depois codificadas com `OneHotEncoder`.

Essa decisão foi melhor do que introduzir categorias artificiais sem sentido porque:

- preserva interpretação clínica
- atende ao requisito de encoding categórico
- demonstra engenharia de atributos com motivação real

## 6.4 Engenharia de atributos

Também foram criadas features numéricas derivadas:

- `area_perimeter_ratio`
- `compactness_to_concavity_ratio`
- `radius_error_pct`
- `worst_radius_delta`

Esses atributos foram introduzidos para:

- capturar relações entre medidas originais
- oferecer sinais adicionais aos modelos lineares
- mostrar uma etapa explícita de feature engineering no pipeline

## 7. Escolha dos Modelos

O projeto precisava treinar pelo menos dois modelos. Foram escolhidos:

- Logistic Regression
- Decision Tree

## 7.1 Por que Logistic Regression

Logistic Regression foi escolhida porque:

- é uma baseline extremamente forte para dados tabulares estruturados
- é simples, estável e rápida
- produz probabilidades diretamente
- é mais fácil de explicar em contexto clínico
- permite interpretar importância relativa por coeficientes

Além disso, em problemas de saúde, modelos mais simples e interpretáveis costumam ser preferíveis em estágios iniciais de validação.

## 7.2 Por que Decision Tree

Decision Tree foi escolhida porque:

- captura relações não lineares
- é intuitiva para explicar regras e divisões
- serve como contraponto à abordagem linear da regressão logística
- atende ao requisito acadêmico de usar um segundo algoritmo distinto

Ela também ajuda a comparar um modelo mais estável e linear contra um modelo baseado em regras.

## 7.3 Por que não começar com modelos mais complexos

Modelos como Random Forest, XGBoost ou redes neurais poderiam atingir ótimo desempenho, mas não foram escolhidos como foco inicial porque:

- o desafio pedia explicitamente modelos clássicos com Scikit-learn
- a explicabilidade em contexto acadêmico e clínico é mais forte com modelos mais simples
- o tamanho do dataset não exige alta complexidade para entregar bom resultado
- a prioridade era uma solução clara, auditável e pronta para produção acadêmica

Em outras palavras, a escolha foi orientada por equilíbrio entre desempenho, transparência e manutenibilidade.

## 8. Estratégia de Treinamento

## 8.1 Divisão treino/teste

Foi utilizado `train_test_split` com:

- `test_size = 0.20`
- `stratify = target`
- `random_state = 42`

Isso foi feito para:

- preservar a proporção entre classes
- garantir reprodutibilidade
- manter uma avaliação hold-out simples e clara

## 8.2 Busca de hiperparâmetros

Foi usado `GridSearchCV` com `StratifiedKFold`.

### Logistic Regression

Foi ajustado:

- `C`

### Decision Tree

Foram ajustados:

- `max_depth`
- `min_samples_leaf`

O `scoring` escolhido para a busca foi `recall`.

Essa escolha foi deliberada: em vez de otimizar o modelo por acurácia global, o pipeline otimiza diretamente o que mais importa para o problema clínico.

## 8.3 Balanceamento de classe

Foi usado `class_weight="balanced"` nos modelos.

Motivo:

- ajuda a reduzir impacto do desbalanceamento entre benigno e maligno
- reforça a atenção aos exemplos da classe mais crítica

## 9. Estratégia de Avaliação

As métricas usadas foram:

- accuracy
- recall
- F1-score
- classification report
- confusion matrix

## 9.1 Por que recall é a métrica principal

Recall mede a proporção de casos malignos corretamente identificados.

Em saúde, essa métrica é central porque:

- falso negativo é um erro especialmente perigoso
- perder um caso maligno pode atrasar conduta médica
- a métrica está mais alinhada ao risco clínico do que accuracy pura

## 9.2 Por que accuracy sozinha não basta

Accuracy pode parecer alta mesmo quando o modelo falha em capturar casos críticos. Em problemas de saúde, isso é insuficiente, porque a distribuição de classes e o custo do erro importam tanto quanto o desempenho agregado.

## 9.3 Por que F1-score foi mantido

F1-score foi mantido porque:

- combina precisão e recall
- ajuda a controlar o custo operacional dos falsos positivos
- evita uma visão enviesada por apenas uma métrica

## 10. Resultado da Seleção do Modelo

Com os dados atuais, o modelo campeão foi `logistic_regression`.

Resultado validado no conjunto de teste:

- Logistic Regression: accuracy `0.9737`, recall `0.9524`, F1-score `0.9639`
- Decision Tree: accuracy `0.9123`, recall `0.8333`, F1-score `0.8750`

O motivo da escolha do campeão foi objetivo:

- maior recall na classe maligna
- em caso de empate, melhor F1-score
- depois, melhor accuracy

Essa regra foi implementada explicitamente para manter a seleção previsível e alinhada ao problema.

## 11. Explicabilidade

## 11.1 Feature importance

Foi gerada feature importance para todos os modelos treinados.

Motivo:

- fornecer visão global dos atributos mais influentes
- facilitar análise comparativa entre candidatos
- permitir documentação dos fatores mais relevantes

## 11.2 SHAP

SHAP foi utilizado no modelo campeão.

A escolha de usar SHAP no campeão, e não em todos os modelos, foi feita porque:

- SHAP pode ser mais custoso computacionalmente
- o objetivo principal da inferência em produção é explicar o modelo realmente servido
- reduz complexidade desnecessária sem perder valor analítico

Foram gerados:

- summary bar plot
- beeswarm plot
- waterfall plot

Além disso, a API retorna os principais contribuidores por predição para tornar a explicação útil também no uso interativo.

## 12. Arquitetura da API

A API foi construída com FastAPI.

Endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`

## 12.1 Por que FastAPI

FastAPI foi escolhida porque:

- possui validação forte via Pydantic
- oferece documentação automática
- tem boa performance para APIs de inferência
- é simples de integrar com modelos Python e Scikit-learn

## 12.2 Separação de responsabilidades

Alguns arquivos importantes:

- `schemas/patient.py`: define o contrato de entrada e valida payloads
- `services/prediction_service.py`: concentra a lógica de inferência e explicação
- `api/routes.py`: define as rotas HTTP
- `core/bootstrap.py`: inicializa e reutiliza o serviço de predição
- `settings.py`: centraliza caminhos e parâmetros globais

Essa estrutura foi escolhida para evitar colocar lógica de negócio diretamente nas rotas.

## 12.3 Validação de entrada

O schema da paciente aceita campos opcionais e valida se pelo menos uma medida foi enviada.

Isso foi feito porque:

- permite payloads incompletos em cenários reais
- evita requisições vazias
- delega imputação ao pipeline já treinado

## 13. Persistência de Artefatos

Os modelos e metadados são persistidos em:

- `artifacts/model_bundle.joblib`
- `artifacts/model_info.json`

Essa decisão foi importante porque:

- evita retrain a cada inicialização
- permite reuso entre API, testes e scripts
- mantém os resultados da execução documentados

O `model_bundle.joblib` guarda:

- pipelines treinados
- nome do modelo campeão
- metadados
- base usada para explicabilidade

## 14. EDA e Relatórios

A camada de `reporting/` gera automaticamente:

- estatísticas descritivas
- resumo de valores ausentes
- matriz de correlação
- distribuição das classes
- histogramas
- boxplots

Essa etapa foi automatizada porque:

- evita análises manuais fora do repositório
- deixa o projeto reprodutível
- gera evidência visual do entendimento dos dados

## 15. Frontend

O frontend foi construído como dashboard estático servido pela própria aplicação.

Ele foi implementado assim porque:

- o desafio não exigia uma SPA complexa
- a solução fica simples de subir e distribuir
- não é necessário manter um segundo pipeline de build
- o foco principal continua sendo o backend e o pipeline de ML

Mais tarde, a interface foi refinada com bibliotecas frontend via CDN:

- Bootstrap
- Bootstrap Icons
- AOS
- SweetAlert2

Isso permitiu melhorar:

- responsividade
- organização visual
- feedback de sucesso e erro
- percepção de produto mais polido

## 16. Qualidade de Engenharia

O projeto foi estruturado para parecer um sistema de engenharia real, e não apenas um notebook transformado em API.

Boas práticas adotadas:

- separação modular por domínio
- scripts dedicados para treino e execução
- testes automatizados
- documentação
- Docker
- serialização de artefatos
- reprodutibilidade via `random_state`

## 17. Testes

Os testes cobrem:

- geração de artefatos
- inicialização da API
- endpoint de health check
- endpoint de model-info
- endpoint de predição

Isso foi feito para garantir que:

- o pipeline realmente produz os arquivos esperados
- a aplicação sobe com o modelo carregado
- o contrato HTTP principal funciona

## 18. Por Que o Projeto Foi Feito Dessa Forma

A solução foi desenhada dessa forma por três motivos centrais:

### 18.1 Clareza técnica

Cada camada tem uma responsabilidade específica. Isso melhora leitura, manutenção e revisão.

### 18.2 Aderência ao problema de saúde

As escolhas de métrica, modelo e explicabilidade foram guiadas pelo contexto clínico, não apenas por conveniência técnica.

### 18.3 Reprodutibilidade

Tudo importante está no repositório:

- carregamento de dados
- treino
- avaliação
- plots
- artefatos
- API
- testes

Isso torna o projeto auditável e demonstrável do início ao fim.

## 19. Principais Trade-offs

Toda arquitetura envolve escolhas.

Trade-offs assumidos:

- usar modelos mais simples em troca de maior interpretabilidade
- usar frontend estático em vez de framework SPA para reduzir complexidade
- usar SHAP apenas no modelo campeão para equilibrar valor e custo computacional
- manter dependências frontend por CDN para acelerar entrega visual

Esses trade-offs foram considerados aceitáveis porque preservam o foco principal do projeto: uma solução de ML explicável, funcional e bem estruturada.

## 20. Evoluções Futuras

Algumas melhorias naturais para próximas versões:

- calibração explícita de probabilidades
- ajuste de limiar com base em curva precision-recall
- inclusão de modelos ensemble para comparação adicional
- autenticação e controle de acesso na API
- observabilidade com logs estruturados e métricas
- versionamento formal de modelos
- persistência de histórico de predições
- testes de carga e testes de contrato
- internacionalização completa entre backend e frontend

## 21. Conclusão

Este projeto foi construído para atender ao desafio de forma completa e tecnicamente defensável.

Ele não é apenas um classificador exposto em HTTP. Ele foi desenhado como um fluxo integrado de engenharia e machine learning, cobrindo:

- dados
- modelagem
- avaliação
- explicabilidade
- serving
- interface
- documentação
- testes

O ponto central das decisões foi equilibrar:

- desempenho
- interpretabilidade
- robustez
- simplicidade operacional

Esse equilíbrio é especialmente importante em aplicações de saúde, onde um sistema tecnicamente sofisticado, mas difícil de explicar ou validar, pode ser menos útil do que uma solução um pouco mais simples, porém auditável e confiável.
