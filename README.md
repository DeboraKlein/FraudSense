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


