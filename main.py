#!/usr/bin/env python3
"""
Atelier 2 : Modularisation du Code avec MLflow
Script principal pour l'exécution du pipeline ML
"""
import argparse
import sys
import mlflow
import mlflow.sklearn
from pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

def main():
    parser = argparse.ArgumentParser(description="Pipeline de prédiction de churn")
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    parser.add_argument("--full-pipeline", action="store_true", help="Pipeline complet")
    parser.add_argument("--train-path", type=str, default="data/raw/churn-bigml-80.csv")
    parser.add_argument("--test-path", type=str, default="data/raw/churn-bigml-20.csv")
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost"],
    )
    parser.add_argument("--model-path", type=str, default="models/churn_model.joblib")
    parser.add_argument("--experiment-name", type=str, default="Churn_Prediction_Pipeline")
    
    args = parser.parse_args()
    
    # Configuration MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(args.experiment_name)

    mlflow.set_tag("version", "v1.0")
    mlflow.set_tag("developer", "Votre_Nom")
    mlflow.set_tag("environment", "production")


    
    if not any([args.prepare, args.train, args.evaluate, args.full_pipeline]):
        parser.print_help()
        return
    
    try:
        if args.full_pipeline:
            print("🚀 Exécution du pipeline complet...")
            run_full_pipeline(args)
            return
        
        if args.prepare:
            print("📊 Préparation des données...")
            X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
                args.train_path, args.test_path
            )
            print("✅ Préparation terminée!")
        
        if args.train:
            print("🤖 Entraînement du modèle...")
            with mlflow.start_run(run_name=f"Training_{args.model_type}"):
                if "X_train" not in locals():
                    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
                        args.train_path, args.test_path
                    )
                
                # Log des paramètres
                mlflow.log_param("model_type", args.model_type)
                mlflow.log_param("train_path", args.train_path)
                mlflow.log_param("test_path", args.test_path)
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_param("n_features", X_train.shape[1])
                
                model = train_model(X_train, y_train, args.model_type)
                save_model(model, args.model_path)
                
                # Log du modèle
                mlflow.sklearn.log_model(model, "model")
        
        if args.evaluate:
            print("📈 Évaluation du modèle...")
            with mlflow.start_run(run_name=f"Evaluation_{args.model_type}"):
                model = load_model(args.model_path)
                if "X_test" not in locals() or "y_test" not in locals():
                    _, X_test, _, y_test, _, _ = prepare_data(
                        args.train_path, args.test_path
                    )
                
                # Log des paramètres
                mlflow.log_param("model_type", args.model_type)
                mlflow.log_param("model_path", args.model_path)
                
                evaluate_model(model, X_test, y_test, f"Modèle {args.model_type}")
                
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)

def run_full_pipeline(args):
    """Exécute le pipeline complet avec tracking MLflow"""
    
    with mlflow.start_run(run_name=f"Full_Pipeline_{args.model_type}") as run:
        print(f"\n🔬 MLflow Run ID: {run.info.run_id}")
        print(f"🔬 Experiment: {args.experiment_name}")
        
        # Log des paramètres généraux
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("train_path", args.train_path)
        mlflow.log_param("test_path", args.test_path)
        mlflow.log_param("pipeline_mode", "full")
        
        print("\n" + "=" * 50)
        print("ÉTAPE 1: PRÉPARATION DES DONNÉES")
        print("=" * 50)
        X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
            args.train_path, args.test_path
        )
        
        # Log des informations sur les données
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("feature_names", str(feature_names))
        mlflow.log_param("class_distribution_train", str(dict(zip(*[y_train.value_counts().index.tolist(), y_train.value_counts().tolist()]))))
        
        print("\n" + "=" * 50)
        print("ÉTAPE 2: ENTRAÎNEMENT DU MODÈLE")
        print("=" * 50)
        model = train_model(X_train, y_train, args.model_type)
        
        # Log des hyperparamètres du modèle
        if hasattr(model, 'get_params'):
            params = model.get_params()
            for param_name, param_value in params.items():
                mlflow.log_param(f"model_{param_name}", param_value)
        
        print("\n" + "=" * 50)
        print("ÉTAPE 3: SAUVEGARDE DU MODÈLE")
        print("=" * 50)
        save_model(model, args.model_path)
        mlflow.log_param("model_save_path", args.model_path)
        
        # Log du modèle dans MLflow
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=f"Churn_{args.model_type}_model"
        )
        
        print("\n" + "=" * 50)
        print("ÉTAPE 4: ÉVALUATION DU MODÈLE")
        print("=" * 50)
        metrics = evaluate_model(model, X_test, y_test, f"Modèle {args.model_type}")
        
        # Log des métriques
        if metrics:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
        
        print("\n🎉 Pipeline exécuté avec succès!")
        print(f"📊 Visualisez les résultats sur MLflow UI: http://127.0.0.1:5000")

if __name__ == "__main__":
    main()