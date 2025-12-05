# pipeline_new.py
"""
Pipeline modular e sklearn-friendly para detecção de fraude.
- Pré-processamento em ColumnTransformer
- Integração com imblearn para oversampling dentro do CV
- Treino comparável entre modelos com StratifiedKFold
- RandomizedSearchCV para tuning (configurável)
- Tuning de threshold via CV (nested thresholds)
- Função de avaliação com bootstrap CI
- Função de explicabilidade com SHAP / permutation importance

Autor: você
Data: ...
"""

import numpy as np
import pandas as pd
import joblib
import json
import time
from typing import Dict, Tuple, Optional, List

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             roc_auc_score, precision_score, recall_score, f1_score)
from sklearn.inspection import permutation_importance

# imblearn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    _has_xgb = True
except Exception:
    XGBClassifier = None
    _has_xgb = False

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    _has_lgb = True
except Exception:
    LGBMClassifier = None
    _has_lgb = False

try:
    from catboost import CatBoostClassifier
    _has_cat = True
except Exception:
    CatBoostClassifier = None
    _has_cat = False

# utility
import warnings
warnings.filterwarnings("ignore")


# -------------------------
# Preprocessor builder
# -------------------------
def build_preprocessor(numeric_cols: List[str],
                       categorical_cols: List[str],
                       numeric_strategy: str = "median",
                       scale: str = "robust") -> ColumnTransformer:
    """
    Retorna um ColumnTransformer com imputação, scaling e encoding.

    Parâmetros
    ----------
    numeric_cols : list of str
        Nomes das colunas numéricas.
    categorical_cols : list of str
        Nomes das colunas categóricas.
    numeric_strategy : {"mean", "median"}, default="median"
        Estratégia de imputação para variáveis numéricas.
    scale : {"robust", "standard", None}, default="robust"
        Tipo de escalonamento a aplicar nas variáveis numéricas.
    """
    from sklearn.pipeline import Pipeline as SkPipeline

    # ----- Pipeline numérico -----
    num_steps = [("imputer", SimpleImputer(strategy=numeric_strategy))]
    if scale == "robust":
        num_steps.append(("scaler", RobustScaler()))
    elif scale == "standard":
        num_steps.append(("scaler", StandardScaler()))
    # se scale for None, não adiciona scaler

    numeric_pipeline = SkPipeline(steps=num_steps)

    # ----- Pipeline categórico (opcional) -----
    if categorical_cols:
        # compatibilidade entre versões do scikit-learn
        try:
            # sklearn >= 1.2
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            # sklearn < 1.2
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        cat_pipeline = SkPipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("ohe", ohe),
        ])
    else:
        cat_pipeline = None

    # ----- Monta o ColumnTransformer -----
    transformers = [
        ("num", numeric_pipeline, numeric_cols),
    ]
    if cat_pipeline is not None:
        transformers.append(("cat", cat_pipeline, categorical_cols))

    preproc = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preproc



# -------------------------
# Model factory
# -------------------------
def get_model(name: str, random_state: int = 42):
    """
    Retorna uma instância de modelo por nome.
    name in ['lr','rf','xgb','lgb','cat']
    """
    name = name.lower()
    if name == "lr":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state, n_jobs=-1)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=random_state)
    if name == "xgb":
        if not _has_xgb:
            raise ImportError("xgboost não encontrado. pip install xgboost")
        return XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=-1, random_state=random_state)
    if name == "lgb":
        if not _has_lgb:
            raise ImportError("lightgbm não encontrado. pip install lightgbm")
        return LGBMClassifier(n_jobs=-1, random_state=random_state)
    if name == "cat":
        if not _has_cat:
            raise ImportError("catboost não encontrado. pip install catboost")
        return CatBoostClassifier(silent=True, random_state=random_state)
    raise ValueError(f"Modelo desconhecido: {name}")


# -------------------------
# Build full pipeline (with optional sampler)
# -------------------------
def build_full_pipeline(preprocessor: ColumnTransformer,
                        model,
                        sampler=None):
    """
    Retorna um imblearn Pipeline com preprocessor -> sampler (opcional) -> model
    sampler: e.g., SMOTE(), RandomOverSampler(), None
    """
    steps = []
    steps.append(("preprocessor", preprocessor))
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("clf", model))
    return ImbPipeline(steps=steps)


# -------------------------
# Cross-validated training for multiple models
# -------------------------
def train_cv_models(X: pd.DataFrame, y: pd.Series,
                    numeric_cols: List[str], categorical_cols: List[str],
                    model_names: List[str] = ["lr","rf","xgb","lgb"],
                    sampler=None,
                    cv_splits: int = 5,
                    scoring: str = "average_precision",
                    random_state: int = 42):
    """
    Treina e compara modelos usando StratifiedKFold CV.
    Retorna um dict com resultados: mean, std, fitted estimators (trained on full train set).
    Note: hyperparameter tuning pode ser feito posteriormente com RandomizedSearchCV.
    """
    results = {}
    preproc = build_preprocessor(numeric_cols, categorical_cols)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    for name in model_names:
        try:
            model = get_model(name, random_state=random_state)
        except Exception as e:
            print(f"Skip model {name}: {e}")
            continue
        pipe = build_full_pipeline(preproc, model, sampler=sampler)
        # cross_val_score uses the pipeline and automatically applies preproc per-fold
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = {"cv_scores": scores, "cv_mean": np.mean(scores), "cv_std": np.std(scores)}
        print(f"Model {name}: {scoring} mean={results[name]['cv_mean']:.4f} std={results[name]['cv_std']:.4f}")
    return results, preproc


# -------------------------
# Randomized search tuning wrapper
# -------------------------
def tune_model_randomized(X: pd.DataFrame, y: pd.Series,
                          preproc: ColumnTransformer,
                          model_name: str,
                          param_distributions: dict,
                          sampler = None,
                          cv_splits: int = 3,
                          n_iter: int = 30,
                          scoring: str = "f1",
                          random_state: int = 42):
    """
    Executa RandomizedSearchCV sobre um pipeline: preproc -> sampler (opt) -> model.
    Retorna o RandomizedSearchCV result (fit).
    """
    model = get_model(model_name, random_state=random_state)
    pipeline = build_full_pipeline(preproc, model, sampler=sampler)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rsearch = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
        return_train_score=False
    )
    t0 = time.time()
    rsearch.fit(X, y)
    t1 = time.time()
    print(f"RandomizedSearchCV finished in {(t1-t0):.1f}s - best_score: {rsearch.best_score_:.4f}")
    return rsearch


# -------------------------
# Threshold tuning with CV
# -------------------------
def nested_threshold_cv(X: pd.DataFrame, y: pd.Series,
                        pipeline,
                        cv_splits: int = 5,
                        beta: float = 1.0,
                        precision_target: Optional[float] = None,
                        random_state: int = 42) -> Dict[str, float]:
    """
    Ajusta o threshold de decisão usando validação cruzada.

    Para cada fold:
    - Treina o pipeline no fold de treino.
    - Calcula probabilidades no fold de validação.
    - Encontra o melhor threshold:
        * Se precision_target for None: maximiza F_beta.
        * Se precision_target for um valor (ex: 0.8): escolhe o menor threshold
          que atinge pelo menos essa precision.
    Retorna:
    - thresholds por fold
    - média e std dos thresholds
    - métricas médias (precision, recall, f1, ap, roc_auc) e seus desvios.
    """

    def f_beta(p: float, r: float, beta: float) -> float:
        denom = (beta**2) * p + r
        return (1 + beta**2) * (p * r) / denom if denom > 0 else 0.0

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    thresholds = []
    fold_metrics = []

    from sklearn.base import clone

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # clone seguro do pipeline
        pipe = clone(pipeline)

        # treinar no fold
        pipe.fit(X_tr, y_tr)

        # probabilidades no fold de validação
        probs = pipe.predict_proba(X_val)[:, 1]
        prec, rec, th = precision_recall_curve(y_val, probs)

        if precision_target is not None:
            # índices onde a precision >= alvo
            valid = np.where(prec >= precision_target)[0]
            if valid.size > 0:
                idx = valid.min()
                # th tem tamanho len(prec)-1
                if idx < len(th):
                    chosen = th[idx]
                else:
                    chosen = th[-1]
            else:
                chosen = 0.5
        else:
            # maximizar F_beta ao longo da curva
            fbs = np.array([f_beta(p, r, beta) for p, r in zip(prec[:-1], rec[:-1])])
            if fbs.size == 0 or np.all(np.isnan(fbs)):
                chosen = 0.5
            else:
                best_idx = int(np.nanargmax(fbs))
                chosen = th[best_idx]

        thresholds.append(chosen)

        # métricas no fold com threshold escolhido
        y_pred = (probs >= chosen).astype(int)
        fold_metrics.append({
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall":    recall_score(y_val, y_pred, zero_division=0),
            "f1":        f1_score(y_val, y_pred, zero_division=0),
            "ap":        average_precision_score(y_val, probs),
            "roc_auc":   roc_auc_score(y_val, probs),
        })

    thresholds = np.array(thresholds)
    metrics_df = pd.DataFrame(fold_metrics)

    return {
        "thresholds_fold": thresholds.tolist(),
        "threshold_mean": float(thresholds.mean()),
        "threshold_std": float(thresholds.std()),
        "metrics_mean": metrics_df.mean().to_dict(),
        "metrics_std": metrics_df.std().to_dict(),
    }



# -------------------------
# Evaluation with bootstrap CI
# -------------------------
def evaluate_with_ci(pipeline, X: pd.DataFrame, y: pd.Series,
                     n_bootstrap: int = 1000, seed: int = 42):
    """
    Avalia modelo e calcula CI bootstrap (95%) para métricas principais
    Retorna: dict com metrics and CI
    """
    probs = pipeline.predict_proba(X)[:, 1]
    base = {
        "ap": average_precision_score(y, probs),
        "roc_auc": roc_auc_score(y, probs)
    }

    rng = np.random.RandomState(seed)
    ap_samples = []
    roc_samples = []
    for i in range(n_bootstrap):
        idx = rng.choice(len(X), size=len(X), replace=True)
        ap_samples.append(average_precision_score(y.iloc[idx], probs[idx]))
        roc_samples.append(roc_auc_score(y.iloc[idx], probs[idx]))

    def ci(arr):
        lo = np.percentile(arr, 2.5)
        hi = np.percentile(arr, 97.5)
        return lo, hi

    ap_lo, ap_hi = ci(ap_samples)
    roc_lo, roc_hi = ci(roc_samples)

    return {
        "ap": base["ap"], "ap_ci": (ap_lo, ap_hi),
        "roc_auc": base["roc_auc"], "roc_auc_ci": (roc_lo, roc_hi)
    }


# -------------------------
# Explainability helpers
# -------------------------
def explain_model_shap(pipeline, X_sample: pd.DataFrame, max_display: int = 20):
    """
    Executa SHAP (TreeExplainer para modelos tree-based) ou KernelExplainer como fallback.
    Retorna explainer e shap_values (pandas-friendly).
    """
    try:
        import shap
    except Exception:
        raise ImportError("Instale shap para usar explicabilidade (pip install shap)")

    # extraia o modelo e o preprocessor
    preproc = pipeline.named_steps.get("preprocessor", None)
    model = pipeline.named_steps.get("clf", None)
    if model is None:
        raise ValueError("Pipeline deve conter um step 'clf' com o modelo.")

    # transformar inputs para o formato exato do modelo
    X_trans = preproc.transform(X_sample)
    feature_names = preproc.get_feature_names_out()

    if hasattr(model, "predict_proba") and ( (hasattr(model, "feature_importances_")) or "tree" in model.__class__.__name__.lower()):
        expl = shap.TreeExplainer(model)
        shap_values = expl.shap_values(X_trans)
        # shap_values shape logic for binary tree models: [n_obs, n_features]
        return expl, shap_values, feature_names
    else:
        expl = shap.KernelExplainer(model.predict_proba, X_trans[:50])
        shap_values = expl.shap_values(X_trans)
        return expl, shap_values, feature_names


# -------------------------
# Permutation importance helper
# -------------------------
def permutation_importances(pipeline, X: pd.DataFrame, y: pd.Series, n_repeats:int=10, random_state:int=42):
    """
    Calcula permutation importance (modelo inteiro: preproc + clf)
    """
    r = permutation_importance(pipeline, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    imp_df = pd.DataFrame({"feature": pipeline.named_steps["preprocessor"].get_feature_names_out(), "importance_mean": r.importances_mean, "importance_std": r.importances_std})
    imp_df = imp_df.sort_values("importance_mean", ascending=False)
    return imp_df


# -------------------------
# Save/load helpers
# -------------------------
def save_pipeline(pipeline, path: str):
    joblib.dump(pipeline, path)
    print(f"Pipeline salvo em {path}")

def load_pipeline(path: str):
    return joblib.load(path)

# -------------------------
# Example: quick train flow (not executed at import)
# -------------------------
if __name__ == "__main__":
    print("Este módulo fornece funções para construir pipeline e treinar modelos com CV.")
    print("Importe as funções e use nos notebooks (recomendado).")
