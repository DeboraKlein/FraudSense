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

## Sobre o Projeto

**FraudSense** é um pipeline completo de *Detecção de Fraude* desenvolvido com foco em **melhores práticas de Machine Learning aplicado ao sistema financeiro**.

O objetivo é detectar transações fraudulentas no dataset altamente desbalanceado do Kaggle [_Credit Card Fraud Detection_], aplicando:

- Pré-processamento profissional com `ColumnTransformer`  
- Balanceamento *somente dentro do Cross-Validation* (evitando data leakage)  
- Comparação justa entre modelos  
- *Nested CV* para tuning de **threshold**  
- Avaliação final em **holdout não visto**  
- Explicabilidade com **SHAP** e **Permutation Importance**  
- Função de **deploy** simulando produção  

O projeto segue rigorosamente o CRISP-DM.

---

# Principais Resultados

### Melhor modelo: **XGBoost**  
- AP (AUC-PR CV): **0.857 ± 0.025**  
- Precision após threshold: **0.95**  
- Recall após threshold: **0.82**  
- Threshold calibrado via nested CV: **~0.995**  

> **Isso reflete exatamente o que fintechs fazem**:  
> Maximizar precisão, manter recall alto e reduzir falsos alertas.

---

# Arquitetura do Projeto

````
FraudSense/
├── notebooks/
│ ├── 01_eda_analysis.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_model_training.ipynb
│ ├── 04_evaluation_deployment.ipynb
│
├── pipeline_new.py # Pipeline unificado do projeto
├── artifacts/
│ ├── preprocessor.joblib
│ ├── best_pipeline.joblib
│ ├── threshold.json
│
├── README.md
└── Glossario.md
````


---

# 🔬 Etapas do Projeto (CRISP-DM)

## **1. Entendimento do Negócio**
Fraudes representam perdas significativas para bancos e fintechs.  
O foco do projeto é **detectar o máximo possível de fraudes**, sem aumentar falsos positivos e sem prejudicar a experiência do usuário.

---

## **2. Entendimento dos Dados**
- 284.807 transações
- Apenas **0,172% são fraude**
- Variáveis V1–V28 já são PCA
- Forte desbalanceamento → cuidado extremo com leakage

---

## **3. Preparação dos Dados**
Criado pipeline com:

- Imputação robusta (`median`)
- Normalização `RobustScaler`
- One-Hot Encoder para categorias futuras
- SMOTE dentro do CV (via `imblearn`)
- *ColumnTransformer* estruturado

Pipeline salvo para reuso em produção.

---

## **4. Modelagem**
Modelos treinados em validação cruzada estratificada:

- Regressão Logística  
- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost  

Métrica principal: **Average Precision (AUC-PR)**  
Justificativa → dataset extremamente desbalanceado.

---

## **5. Avaliação**
Inclui:

- Holdout final nunca visto  
- Curva Precision-Recall  
- Curva ROC  
- Matriz de Confusão  
- Threshold tuning via nested CV  
- Explicabilidade com SHAP  
- Permutation Importance  

---

## **6. Deploy Simulado**
Função final:

```python
def predict_transactions(pipeline, df, threshold):
    probs = pipeline.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)
    return preds, probs

### Como Reproduzir

1. Instale dependências
````
pip install -r requirements.txt
````

### 2. Rode os notebooks na ordem:

    01_eda_analysis.ipynb

    02_preprocessing.ipynb

    03_model_training.ipynb

    04_evaluation_deployment.ipynb

### 3. Execute pipeline_new.py para importar funções centrais.

## Explicabilidade (SHAP)

    Summary Plot global

    Waterfall plot de uma transação fraudulenta

    Permutation Importance

    Análise de quais features puxam risco para cima ou para baixo

Essencial para auditoria e uso em instituições financeiras.

##  Próximos Passos

    Implementar API REST (FastAPI)

    Monitoramento de drift

    Ajuste dinâmico de threshold

    Integração com simulação de regra de negócio

##  Autora

Projeto desenvolvido por Debora Rebula como estudo avançado em ML para sistemas antifraude.

## Licença

    MIT — livre para uso e adaptação.


