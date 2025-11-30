"""
Atelier 2 : Modularisation du Code
Pipeline modulaire pour la prédiction de churn
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss,
    cohen_kappa_score,
    matthews_corrcoef,
)
from imblearn.combine import SMOTEENN
import joblib
import os

# Configuration
CONFIG = {
    "TRAIN_DATA_PATH": "data/raw/churn-bigml-80.csv",
    "TEST_DATA_PATH": "data/raw/churn-bigml-20.csv",
    "MODEL_PATH": "models/churn_model.joblib",
    "SCALER_PATH": "models/scaler.joblib",
    "ENCODER_STATE_PATH": "models/encoder_state.joblib",
    "ENCODER_AREA_PATH": "models/encoder_area.joblib",
    "RANDOM_STATE": 42,
    "SMOTE_SAMPLING_RATIO": 30 / 70,
    "Z_SCORE_THRESHOLD": 3,
    "IQR_MULTIPLIER": 1.5,
}


def prepare_data(train_path=None, test_path=None):
    """
    Charger et prétraiter les données pour l'entraînement.
    """
    if train_path is None:
        train_path = CONFIG["TRAIN_DATA_PATH"]
    if test_path is None:
        test_path = CONFIG["TEST_DATA_PATH"]

    print("Étape 1: Chargement des données...")
    X_train, X_test = _load_and_combine_data(train_path, test_path)

    print("Étape 2: Encodage des features catégorielles...")
    X_train, X_test, encoder_state, encoder_area = _encode_categorical_features(
        X_train, X_test
    )

    print("Étape 3: Gestion des outliers...")
    X_train = _handle_outliers(X_train)

    print("Étape 4: Feature engineering...")
    X_train = _create_engineered_features(X_train)
    X_test = _create_engineered_features(X_test)

    print("Étape 5: Suppression des features corrélées...")
    X_train = _drop_correlated_features(X_train)
    X_test = _drop_correlated_features(X_test)

    print("Étape 6: Séparation features/target...")
    y_train = X_train["Churn"]
    X_train = X_train.drop(["Churn"], axis=1)
    y_test = X_test["Churn"]
    X_test = X_test.drop(["Churn"], axis=1)

    print("Étape 7: Équilibrage des données...")
    X_train, y_train = _balance_data(X_train, y_train)

    print("Étape 8: Normalisation des features...")
    X_train_scaled, X_test_scaled, scaler = _scale_features(X_train, X_test)

    _save_preprocessors(scaler, encoder_state, encoder_area)

    joblib.dump(X_train.columns.tolist(), "models/columns_order.joblib")

    print(f"Préparation terminée. Shape final: {X_train_scaled.shape}")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        X_train.columns.tolist(),
    )


def train_model(X_train, y_train, model_type="random_forest", **params):
    """
    Entraîner un modèle de machine learning.
    """
    print(f"Entraînement du modèle {model_type}...")

    if model_type == "random_forest":
        model = _train_random_forest(X_train, y_train, params)
    elif model_type == "xgboost":
        model = _train_xgboost(X_train, y_train, params)
    else:
        raise ValueError("Type de modèle non supporté")

    print("Entraînement terminé avec succès!")
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Évaluer les performances du modèle.
    
    Returns:
        dict: Métriques d'évaluation
    """
    print(f"Évaluation du modèle {model_name}...")
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    logloss = log_loss(y_test, model.predict_proba(X_test))
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # Métriques détaillées
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'log_loss': logloss,
        'cohen_kappa': kappa,
        'matthews_corrcoef': mcc,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    
    # Affichage du rapport
    _print_evaluation_report(metrics)
    
    # RETOURNER LES MÉTRIQUES - C'EST IMPORTANT !
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'log_loss': logloss,
        'cohen_kappa': kappa,
        'matthews_corrcoef': mcc
    }


def save_model(model, filepath=None):
    """
    Sauvegarder le modèle entraîné.
    """
    if filepath is None:
        filepath = CONFIG["MODEL_PATH"]

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Modèle sauvegardé: {filepath}")


def load_model(filepath=None):
    """
    Charger un modèle sauvegardé.
    """
    if filepath is None:
        filepath = CONFIG["MODEL_PATH"]

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Modèle non trouvé: {filepath}")

    model = joblib.load(filepath)
    print(f"Modèle chargé: {filepath}")
    return model


# =============================================================================
# FONCTIONS INTERNES
# =============================================================================


def _load_and_combine_data(train_path, test_path):
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)
    return X_train, X_test


def _encode_categorical_features(X_train, X_test):
    binary_cols = ["International plan", "Voice mail plan"]

    for col in binary_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].map({"No": 0, "Yes": 1})
            X_test[col] = X_test[col].map({"No": 0, "Yes": 1})

    if "Churn" in X_train.columns:
        X_train["Churn"] = X_train["Churn"].astype(int)
    if "Churn" in X_test.columns:
        X_test["Churn"] = X_test["Churn"].astype(int)

    encoder_state = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder_area = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    encoded_states_train = encoder_state.fit_transform(X_train[["State"]])
    encoded_states_test = encoder_state.transform(X_test[["State"]])

    encoded_area_train = encoder_area.fit_transform(X_train[["Area code"]])
    encoded_area_test = encoder_area.transform(X_test[["Area code"]])

    encoded_states_df_train = pd.DataFrame(
        encoded_states_train, columns=encoder_state.get_feature_names_out(["State"])
    )
    encoded_states_df_test = pd.DataFrame(
        encoded_states_test, columns=encoder_state.get_feature_names_out(["State"])
    )

    encoded_area_df_train = pd.DataFrame(
        encoded_area_train, columns=encoder_area.get_feature_names_out(["Area code"])
    )
    encoded_area_df_test = pd.DataFrame(
        encoded_area_test, columns=encoder_area.get_feature_names_out(["Area code"])
    )

    X_train = X_train.drop(["State", "Area code"], axis=1)
    X_test = X_test.drop(["State", "Area code"], axis=1)

    X_train = pd.concat(
        [X_train, encoded_states_df_train, encoded_area_df_train], axis=1
    )
    X_test = pd.concat([X_test, encoded_states_df_test, encoded_area_df_test], axis=1)

    return X_train, X_test, encoder_state, encoder_area


def _handle_outliers(df):
    numerical_cols = [
        "Account length",
        "Total day minutes",
        "Total day calls",
        "Total day charge",
        "Total eve minutes",
        "Total eve calls",
        "Total eve charge",
        "Total night minutes",
        "Total night calls",
        "Total night charge",
        "Total intl minutes",
        "Total intl calls",
        "Total intl charge",
    ]

    from scipy.stats import zscore

    z_scores = zscore(df[numerical_cols])
    filtered_entries = (np.abs(z_scores) < CONFIG["Z_SCORE_THRESHOLD"]).all(axis=1)
    return df[filtered_entries]


def _create_engineered_features(df):
    df["Total calls"] = (
        df["Total day calls"]
        + df["Total eve calls"]
        + df["Total night calls"]
        + df["Total intl calls"]
    )

    df["Total charge"] = (
        df["Total day charge"]
        + df["Total eve charge"]
        + df["Total night charge"]
        + df["Total intl charge"]
    )

    df["CScalls Rate"] = df["Customer service calls"] / (df["Account length"] + 1)
    return df


def _drop_correlated_features(df):
    correlated_cols = [
        "Total day minutes",
        "Total eve minutes",
        "Total night minutes",
        "Total intl minutes",
        "Voice mail plan",
    ]

    return df.drop(columns=[col for col in correlated_cols if col in df.columns])


def _balance_data(X, y):
    smote_enn = SMOTEENN(
        sampling_strategy=CONFIG["SMOTE_SAMPLING_RATIO"],
        random_state=CONFIG["RANDOM_STATE"],
    )
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    return X_resampled, y_resampled


def _scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def _save_preprocessors(scaler, encoder_state, encoder_area):
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, CONFIG["SCALER_PATH"])
    joblib.dump(encoder_state, CONFIG["ENCODER_STATE_PATH"])
    joblib.dump(encoder_area, CONFIG["ENCODER_AREA_PATH"])
    print("Préprocesseurs sauvegardés!")


def _train_random_forest(X_train, y_train, params):
    default_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": CONFIG["RANDOM_STATE"],
        "class_weight": "balanced",
    }
    default_params.update(params)

    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def _train_xgboost(X_train, y_train, params):
    try:
        from xgboost import XGBClassifier

        default_params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "random_state": CONFIG["RANDOM_STATE"],
        }
        default_params.update(params)

        model = XGBClassifier(**default_params)
        model.fit(X_train, y_train)
        return model
    except ImportError:
        raise ImportError("XGBoost n'est pas installé")


def _print_evaluation_report(metrics):
    print(f"\n{'='*60}")
    print(f"{metrics['model_name']} - Rapport d'Évaluation")
    print(f"{'='*60}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    print(f"Matthews Correlation Coefficient: {metrics['matthews_corrcoef']:.4f}")
    print(f"\nMatrice de Confusion:\n{metrics['confusion_matrix']}")
    print(f"\nRapport de Classification:\n{metrics['classification_report']}")
    print(f"{'='*60}\n")
