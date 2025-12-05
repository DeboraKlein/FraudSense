# fraud_pipeline.py
"""
FraudSense - Pipeline de Detecção de Fraudes em Produção
Autor: [Seu Nome]
Data: [Data]
Descrição: Pipeline completo para detecção de fraudes usando XGBoost otimizado
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, auc,
                           recall_score, precision_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionPipeline:
    """
    Pipeline completo para detecção de fraudes em transações financeiras
    """
    
    def __init__(self, model_path=None):
        """
        Inicializa o pipeline
        
        Args:
            model_path (str): Caminho para modelo pré-treinado (opcional)
        """
        self.model = None
        self.scaler_time = RobustScaler()
        self.scaler_amount = RobustScaler()
        self.selector = None
        self.feature_names = None
        self.is_trained = False
        
        # Carregar modelo se fornecido
        if model_path:
            self.load_model(model_path)
    
    def load_data(self, file_path):
        """
        Carrega dados do dataset de fraudes
        
        Args:
            file_path (str): Caminho para o arquivo CSV
            
        Returns:
            pandas.DataFrame: Dataset carregado
        """
        print(" Carregando dados...")
        df = pd.read_csv(file_path)
        print(f" Dados carregados: {df.shape[0]:,} transações, {df.shape[1]} features")
        return df
    
    def preprocess_data(self, df, training=True):
        """
        Pré-processa os dados para treinamento ou predição
        
        Args:
            df (DataFrame): Dados brutos
            training (bool): Se True, ajusta scalers e selector
            
        Returns:
            tuple: (X_processed, y) para treino ou X_processed para predição
        """
        print(" Pré-processando dados...")
        
        # Separar features e target
        if 'Class' in df.columns:
            X = df.drop('Class', axis=1)
            y = df['Class']
        else:
            X = df.copy()
            y = None
        
        # Features originais para escalonamento
        original_features = ['Time', 'Amount']
        pca_features = [f'V{i}' for i in range(1, 29)]
        
        # Criar cópia para não modificar o original
        X_processed = X.copy()
        
        if training:
            # Ajustar scalers nos dados de treino
            X_processed['Time'] = self.scaler_time.fit_transform(X[['Time']])
            X_processed['Amount'] = self.scaler_amount.fit_transform(X[['Amount']])
            
            # Seleção de features
            self.selector = SelectKBest(score_func=f_classif, k=20)
            X_selected = self.selector.fit_transform(X_processed, y)
            self.feature_names = X_processed.columns[self.selector.get_support()].tolist()
            
            print(f" Pré-processamento concluído: {X_selected.shape[1]} features selecionadas")
            return X_selected, y
            
        else:
            # Usar scalers já ajustados para predição
            X_processed['Time'] = self.scaler_time.transform(X[['Time']])
            X_processed['Amount'] = self.scaler_amount.transform(X[['Amount']])
            
            # Aplicar seleção de features
            if self.selector is not None:
                X_selected = self.selector.transform(X_processed)
                print(f" Pré-processamento concluído: {X_selected.shape[1]} features")
                return X_selected
            else:
                raise ValueError("Pipeline não treinado. Execute o treinamento primeiro.")
    
    def train(self, file_path, test_size=0.2, random_state=42):
        """
        Treina o pipeline completo
        
        Args:
            file_path (str): Caminho para os dados de treino
            test_size (float): Proporção para conjunto de teste
            random_state (int): Seed para reprodutibilidade
            
        Returns:
            dict: Métricas de avaliação
        """
        print(" INICIANDO TREINAMENTO DO PIPELINE...")
        
        # 1. Carregar dados
        df = self.load_data(file_path)
        
        # 2. Divisão estratificada
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop('Class', axis=1), df['Class'],
            test_size=test_size,
            stratify=df['Class'],
            random_state=random_state
        )
        
        print(f" Divisão dos dados:")
        print(f"   • Treino: {X_train.shape[0]:,} amostras")
        print(f"   • Teste:  {X_test.shape[0]:,} amostras")
        
        # 3. Pré-processamento
        X_train_processed, y_train = self.preprocess_data(
            pd.concat([X_train, y_train], axis=1), training=True
        )
        
        # 4. Balanceamento com SMOTE
        print(" Aplicando SMOTE para balanceamento...")
        smote = SMOTE(random_state=random_state)
        X_balanced, y_balanced = smote.fit_resample(X_train_processed, y_train)
        
        print(f" Balanceamento:")
        print(f"   • Antes: {y_train.sum():,} fraudes ({y_train.mean():.4%})")
        print(f"   • Depois: {y_balanced.sum():,} fraudes ({y_balanced.mean():.4%})")
        
        # 5. Treinar modelo XGBoost otimizado
        print(" Treinando modelo XGBoost...")
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=25,
            random_state=random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        self.model.fit(X_balanced, y_balanced)
        self.is_trained = True
        
        print(" Modelo treinado com sucesso!")
        
        # 6. Avaliar no conjunto de teste
        X_test_processed = self.preprocess_data(X_test, training=False)
        metrics = self.evaluate(X_test_processed, y_test)
        
        return metrics
    
    def evaluate(self, X, y):
        """
        Avalia o modelo nos dados fornecidos
        
        Args:
            X (array): Features
            y (array): Target
            
        Returns:
            dict: Métricas de avaliação
        """
        if not self.is_trained:
            raise ValueError("Pipeline não treinado. Execute o treinamento primeiro.")
        
        print(" Avaliando modelo...")
        
        # Previsões
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Métricas
        accuracy = (y_pred == y).mean()
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0
        auc_roc = roc_auc_score(y, y_pred_proba)
        
        # AUC Precision-Recall
        precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred_proba)
        auc_pr = auc(recall_curve, precision_curve)
        
        # Matriz de confusão
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'f2_score': f2,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'confusion_matrix': cm,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        }
        
        # Print resultados
        print("\n" + "="*50)
        print(" RESULTADOS DA AVALIAÇÃO")
        print("="*50)
        print(f" Métricas Principais:")
        print(f"   • Recall:    {recall:.4f}")
        print(f"   • Precision: {precision:.4f}")
        print(f"   • F2-Score:  {f2:.4f}")
        print(f"   • AUC-PR:    {auc_pr:.4f}")
        
        print(f"\n Matriz de Confusão:")
        print(f"               Previsto")
        print(f"             0       1")
        print(f"Real  0   [{tn:>6} {fp:>6}]")
        print(f"      1   [{fn:>6} {tp:>6}]")
        
        print(f"\n Impacto no Negócio:")
        print(f"   • Fraudes detectadas: {tp}/{tp+fn} ({tp/(tp+fn):.1%})")
        print(f"   • Falsos positivos: {fp} transações legítimas bloqueadas")
        print(f"   • Falsos negativos: {fn} fraudes não detectadas")
        
        return metrics
    
    def predict(self, X, return_proba=False, threshold=0.5):
        """
        Faz previsões para novos dados
        
        Args:
            X (DataFrame ou array): Dados para predição
            return_proba (bool): Se True, retorna probabilidades
            threshold (float): Limiar para classificação
            
        Returns:
            array: Previsões (0 ou 1) ou probabilidades
        """
        if not self.is_trained:
            raise ValueError("Pipeline não treinado. Execute o treinamento primeiro.")
        
        # Pré-processar dados
        if isinstance(X, pd.DataFrame):
            X_processed = self.preprocess_data(X, training=False)
        else:
            # Assumir que já está pré-processado
            X_processed = X
        
        # Fazer previsões
        if return_proba:
            return self.model.predict_proba(X_processed)[:, 1]
        else:
            probas = self.model.predict_proba(X_processed)[:, 1]
            return (probas >= threshold).astype(int)
    
    def predict_single(self, transaction_data):
        """
        Faz previsão para uma única transação
        
        Args:
            transaction_data (dict): Dados da transação
            
        Returns:
            dict: Resultado da predição com explicação
        """
        if not self.is_trained:
            raise ValueError("Pipeline não treinado. Execute o treinamento primeiro.")
        
        # Converter para DataFrame
        df_single = pd.DataFrame([transaction_data])
        
        # Pré-processar
        X_processed = self.preprocess_data(df_single, training=False)
        
        # Previsão
        fraud_probability = self.model.predict_proba(X_processed)[:, 1][0]
        is_fraud = fraud_probability >= 0.5
        
        # Feature importance (se disponível)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        else:
            top_features = []
        
        result = {
            'is_fraud': bool(is_fraud),
            'fraud_probability': float(fraud_probability),
            'threshold_used': 0.5,
            'top_features': top_features,
            'confidence': 'HIGH' if fraud_probability > 0.8 else 'MEDIUM' if fraud_probability > 0.5 else 'LOW'
        }
        
        return result
    
    def save_model(self, file_path):
        """
        Salva o pipeline treinado
        
        Args:
            file_path (str): Caminho para salvar o modelo
        """
        if not self.is_trained:
            raise ValueError("Nenhum modelo treinado para salvar.")
        
        model_package = {
            'model': self.model,
            'scaler_time': self.scaler_time,
            'scaler_amount': self.scaler_amount,
            'selector': self.selector,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_package, file_path)
        print(f" Pipeline salvo em: {file_path}")
    
    def load_model(self, file_path):
        """
        Carrega um pipeline treinado
        
        Args:
            file_path (str): Caminho para o modelo salvo
        """
        model_package = joblib.load(file_path)
        
        self.model = model_package['model']
        self.scaler_time = model_package['scaler_time']
        self.scaler_amount = model_package['scaler_amount']
        self.selector = model_package['selector']
        self.feature_names = model_package['feature_names']
        self.is_trained = model_package['is_trained']
        
        print(f" Pipeline carregado de: {file_path}")


# Funções de utilidade para uso direto
def create_and_train_pipeline(data_path, model_save_path='fraud_pipeline.pkl'):
    """
    Função conveniente para criar e treinar o pipeline
    
    Args:
        data_path (str): Caminho para os dados
        model_save_path (str): Caminho para salvar o modelo
        
    Returns:
        FraudDetectionPipeline: Pipeline treinado
    """
    pipeline = FraudDetectionPipeline()
    metrics = pipeline.train(data_path)
    pipeline.save_model(model_save_path)
    return pipeline, metrics

def load_existing_pipeline(model_path):
    """
    Função conveniente para carregar pipeline existente
    
    Args:
        model_path (str): Caminho para o modelo salvo
        
    Returns:
        FraudDetectionPipeline: Pipeline carregado
    """
    return FraudDetectionPipeline(model_path=model_path)


# Exemplo de uso
if __name__ == "__main__":
    """
    Exemplo de como usar o pipeline
    """
    
    print(" FRAUD DETECTION PIPELINE - EXEMPLO DE USO")
    print("="*50)
    
    # Exemplo 1: Treinar novo modelo
    print("\n1.  TREINANDO NOVO MODELO:")
    try:
        pipeline, metrics = create_and_train_pipeline(
            data_path='creditcard.csv',
            model_save_path='fraud_model_production.pkl'
        )
    except Exception as e:
        print(f" Erro no treinamento: {e}")
        print(" Executando em modo de demonstração...")
        
        # Modo demonstração - criar pipeline vazio
        pipeline = FraudDetectionPipeline()
        print(" Pipeline criado (modo demonstração)")
    
    # Exemplo 2: Fazer predição única
    print("\n2.  EXEMPLO DE PREDIÇÃO ÚNICA:")
    
    # Transação de exemplo (valores fictícios)
    example_transaction = {
        'Time': 50000,
        'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536347, 'V4': 1.378155,
        'V5': -0.338321, 'V6': 0.462388, 'V7': 0.239599, 'V8': 0.098698,
        'V9': 0.363787, 'V10': 0.090794, 'V11': -0.551600, 'V12': -0.617801,
        'V13': -0.991390, 'V14': -0.311169, 'V15': 1.468177, 'V16': -0.470401,
        'V17': 0.207971, 'V18': 0.025791, 'V19': 0.403993, 'V20': 0.251412,
        'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474, 'V24': 0.066928,
        'V25': 0.128539, 'V26': -0.189115, 'V27': 0.133558, 'V28': -0.021053,
        'Amount': 149.62
    }
    
    if pipeline.is_trained:
        try:
            result = pipeline.predict_single(example_transaction)
            print(f" Resultado da análise:")
            print(f"   • É fraude: {result['is_fraud']}")
            print(f"   • Probabilidade: {result['fraud_probability']:.3f}")
            print(f"   • Confiança: {result['confidence']}")
            print(f"   • Features importantes: {result['top_features'][:3]}")
        except Exception as e:
            print(f" Erro na predição: {e}")
    else:
        print(" Pipeline não treinado - pulando exemplo de predição")
    
    print("\n PIPELINE PRONTO PARA USO!")
    print(" Use: pipeline = FraudDetectionPipeline('fraud_model_production.pkl')")