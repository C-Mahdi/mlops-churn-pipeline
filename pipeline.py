"""
Module de pipeline ML pour la prédiction de churn
Avec support MLflow pour le tracking des expériences
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import xgboost as xgb
import joblib
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def prepare_data(train_path, test_path):
    """
    Prépare les données d'entraînement et de test
    
    Args:
        train_path: Chemin vers les données d'entraînement
        test_path: Chemin vers les données de test
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    print(f"📂 Chargement des données...")
    print(f"   Training: {train_path}")
    print(f"   Test: {test_path}")
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    print(f"✓ Données chargées: {len(train_data)} train, {len(test_data)} test")
    
    # Séparer features et target
    X_train = train_data.drop("Churn", axis=1)
    y_train = train_data["Churn"]
    X_test = test_data.drop("Churn", axis=1)
    y_test = test_data["Churn"]
    
    # Encoder les variables catégorielles
    print("🔄 Encodage des variables catégorielles...")
    categorical_cols = X_train.select_dtypes(include=["object", "bool"]).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    
    # Encoder la target
    le_target = LabelEncoder()
    y_train = pd.Series(le_target.fit_transform(y_train), name="Churn")
    y_test = pd.Series(le_target.transform(y_test), name="Churn")
    
    # Normalisation
    print("📏 Normalisation des features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir en DataFrame
    feature_names = X_train.columns.tolist()
    X_train = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    print(f"✅ Préparation terminée!")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Train samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, scaler, feature_names

def train_model(X_train, y_train, model_type="random_forest"):
    """
    Entraîne un modèle de classification
    
    Args:
        X_train: Features d'entraînement
        y_train: Target d'entraînement
        model_type: Type de modèle ('random_forest' ou 'xgboost')
    
    Returns:
        model: Modèle entraîné
    """
    print(f"🤖 Entraînement du modèle {model_type}...")
    
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")
    
    model.fit(X_train, y_train)
    
    print(f"✅ Modèle {model_type} entraîné!")
    
    # Log des hyperparamètres dans MLflow (si dans un run actif)
    if mlflow.active_run():
        params = model.get_params()
        for param_name, param_value in params.items():
            try:
                mlflow.log_param(f"{model_type}_{param_name}", param_value)
            except:
                pass  # Ignore si déjà loggé
    
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Évalue les performances du modèle
    
    Args:
        model: Modèle entraîné
        X_test: Features de test
        y_test: Target de test
        model_name: Nom du modèle pour l'affichage
    
    Returns:
        dict: Dictionnaire contenant les métriques
    """
    print(f"📊 Évaluation du {model_name}...")
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }
    
    # Affichage des métriques
    print(f"\n{'='*50}")
    print(f"RÉSULTATS - {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"{'='*50}\n")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion:")
    print(cm)
    print()
    
    # Rapport de classification
    print("Rapport de classification:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Log des métriques dans MLflow (si dans un run actif)
    if mlflow.active_run():
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Créer et logger la matrice de confusion
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Sauvegarder et logger
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)    
        plt.close()
        
        # Courbe ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        roc_path = "roc_curve.png"
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path)
        plt.close()
        
        # Feature importance (si disponible)
        if hasattr(model, 'feature_importances_'):
            # Obtenir les noms des features
            if hasattr(X_test, 'columns'):
                feature_names = X_test.columns.tolist()
            else:
                feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]
            
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title(f'Top 10 Feature Importances - {model_name}')
            plt.tight_layout()
            
            fi_path = "feature_importance.png"
            plt.savefig(fi_path)
            mlflow.log_artifact(fi_path)
            plt.close()
    
    return metrics

def save_model(model, filepath):
    """
    Sauvegarde le modèle
    
    Args:
        model: Modèle à sauvegarder
        filepath: Chemin de sauvegarde
    """
    print(f"💾 Sauvegarde du modèle: {filepath}")
    
    # Créer le dossier si nécessaire
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, filepath)
    print(f"✅ Modèle sauvegardé!")

def load_model(filepath):
    """
    Charge un modèle sauvegardé
    
    Args:
        filepath: Chemin du modèle
    
    Returns:
        model: Modèle chargé
    """
    print(f"📂 Chargement du modèle: {filepath}")
    model = joblib.load(filepath)
    print(f"✅ Modèle chargé!")
    return model