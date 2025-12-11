<div align="center">

# FraudSense — Sistema Inteligente de Detecção de Fraude em Transações Bancárias  
### Pipeline Completo • Validação Robusta • Threshold Calibrado • Explicabilidade SHAP

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Fraud%20Detection-purple)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue?logo=kaggle)

</div>

---



### 1. Visão Geral

O FraudSense é um sistema completo de detecção de transações fraudulentas baseado em Machine Learning, construído com foco em:

   - Reprodutibilidade

   - Robustez estatística

   - Prevenção de leakage

   - Alta precisão operacional

   - Explicabilidade (SHAP e Permutation Importance)

Este projeto segue integralmente a metodologia CRISP-DM, abrangendo desde análise exploratória até avaliação final em holdout e criação de função de deploy.

O objetivo é simular um pipeline real de risco e antifraude utilizado em bancos e fintechs, empregando práticas profissionais de modelagem supervisionada em cenários com alto desbalanceamento.

---

### 2. Motivação

Transações fraudulentas representam um risco financeiro e operacional significativo.
No dataset utilizado (Kaggle – "Credit Card Fraud Detection"):

   - Fraude ≈ 0,17% das transações

   - Forte assimetria entre classes

   - Variáveis anonimizadas via PCA

   - Valores monetários altamente assimétricos

Modelos tradicionais treinados sem cuidados tendem a prever sempre "não fraude" e obter acurácia artificialmente alta.
Neste contexto, o FraudSense é construído para maximizar recall, precision e confiabilidade estatística.

---

### 3. Arquitetura do Projeto

O projeto é estruturado em quatro notebooks, cada um representando uma fase clara do CRISP-DM:

#### Notebook 1 — Análise Exploratória (EDA)

   - Entendimento do desbalanceamento

   - Distribuição de Amount, Time e PCA components

   - Estatísticas descritivas e percentis

   - Identificação de riscos de leakage

   - Construção de narrativa analítica


#### Notebook 2 — Pré-processamento

   - Definição oficial de features

   - Imputação (mediana)

   - Escalonamento robusto (RobustScaler)

   - Criação do preprocessor.joblib

   - Salvamento de metadados (preprocessing_metadata.json)

   - Base para modelagem e deploy

#### Notebook 3 — Modelagem e Tuning

   - Benchmark com 5 modelos (LR, RF, XGB, LGBM, CatBoost)

   - Validação cruzada estratificada com SMOTE dentro dos folds

   - Seleção baseada em Average Precision (AUC-PR)

   - RandomizedSearchCV para hiperparâmetros

   - Tuning de threshold via Nested CV

   - Geração do pipeline final: best_pipeline.joblib

   - Salvamento de threshold.json

#### Notebook 4 — Avaliação Final, Explicabilidade e Deploy

   - Avaliação em holdout (test split nunca visto)

   - Métricas finais: precision, recall, F1, AUC-PR, ROC AUC

   - Matriz de confusão

   - Curvas ROC e PR

   - Explicabilidade: SHAP (global e local) + Permutation Importance

   - Função de deploy predict_transactions()

---

### 4. Técnicas Utilizadas
##### Modelos

   - Logistic Regression

   - Random Forest

   - XGBoost (modelo final)

   - LightGBM

   - CatBoost

##### Métodos Chave

   - SMOTE aplicado somente dentro da cross-validation

   - Grid/Random Search estruturado via pipeline

   - Nested CV para threshold tuning

   - RobustScaler para variáveis financeiras

   - SHAP para explicabilidade individual e global

---

### 5. Resultados Principais

Após ajuste do threshold final (~0.995), o modelo atinge no holdout:

   - Precision: ~0.95

   - Recall: ~0.81

   - F1: ~0.87

   - AUC-PR: ~0.88

   - ROC AUC: ~0.98

Matriz de confusão (holdout):

   - Falsos Positivos (FP): 4

   - Falsos Negativos (FN): 19

   - Fraudes detectadas: 79 de 98

Esses resultados são realistas para um cenário de fraudes financeiras:

   - Alta qualidade dos alertas (precision elevada)

   - Cobertura significativa das fraudes (recall acima de 80%)

   - Threshold conservador, alinhado a operações antifraude reais

---

### 6. Estrutura de Artefatos

````
artifacts/
│── preprocessor.joblib
│── best_pipeline.joblib
│── threshold.json
````

##### Esses arquivos possibilitam:

   - Reproducibilidade total

   - Carregamento direto em APIs ou sistemas de scoring

   - Manutenção de consistência entre ambientes

---

### 7. Função de Deploy

A função abaixo simula o uso do modelo em produção:
````
def predict_transactions(pipeline, df, threshold):
    probs = pipeline.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)
    return preds, probs
````

##### Ela recebe transações brutas e retorna:

   - Probabilidade de fraude

   - Classificação binária

   - Comportamento idêntico ao modelo lançado em produção

---

### 8. Explicabilidade
#### SHAP

   - Summary plot mostra variáveis que mais aumentam ou reduzem o risco.

   - Waterfall plot explica decisões individuais.

#### Permutation Importance

   - Mede a real importância das variáveis ao embaralhar seus valores.

   - Ambos aumentam confiança no modelo para áreas como Risco, Compliance e Auditoria.

### 9. Requisitos

   - Python 3.9+

   - scikit-learn

   - imbalanced-learn

   - XGBoost, LightGBM, CatBoost

   - SHAP

   - joblib

   - seaborn, matplotlib

   - kagglehub

### 10. Estrutura do Repositório

````
FraudSense/
│── notebooks/
│   ├── 01_eda_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation_deployment.ipynb
│
│── pipeline_new.py
│── Glossario.md
│── README.md
│── artifacts/
````

### Lições Aprendidas

Reflexões profissionais e técnicas sobre o desenvolvimento do projeto

---
#### 1. A importância de evitar leakage

Percebi que o maior risco em projetos reais de fraude não é escolher o melhor algoritmo, mas sim evitar vazamento de informação.
Aprendi a:

Colocar SMOTE dentro dos folds

Embutir todo pré-processamento no pipeline

Manter um holdout realmente isolado

Essas práticas aumentam drasticamente a confiabilidade do modelo.

---
#### 2. AUC-ROC não é suficiente

Entendi na prática que modelos com AUC-ROC alto podem ser inúteis em dados altamente desbalanceados.
A métrica AUC-PR representa muito mais fielmente a capacidade do modelo.

---
#### 3. Threshold importa tanto quanto o modelo

Percebi que 90% da performance operacional vem da escolha do threshold, não do algoritmo em si.
Aprender a calibrar o threshold via nested CV foi um ponto-chave.

---
#### 4. Balanceamento deve ser feito com cuidado

SMOTE aplicado fora da validação cruzada gera métricas irreais.
Aprendi a aplicar balanceamento apenas no treinamento de cada fold, garantindo integridade estatística.

---
#### 5. Explicabilidade é indispensável

Ao trabalhar com SHAP e permutation importance, entendi como justificar decisões do modelo para stakeholders de risco e compliance.
Explicabilidade deixou de ser opcional e passou a ser parte fundamental do projeto.

---
#### 6. Modelos tree-based exigem tuning cuidadoso

Percebi como hiperparâmetros influenciam modelos como XGBoost e LightGBM, especialmente em classes desbalanceadas.
Aprendi a:

Ajustar scale_pos_weight

Controlar profundidade

Balancear subsample e colsample

---
#### 7. ML real exige modularidade e artefatos

Criar preprocessor.joblib, best_pipeline.joblib e threshold.json ensinou a pensar como MLOps, e não apenas como modelagem offline.

---