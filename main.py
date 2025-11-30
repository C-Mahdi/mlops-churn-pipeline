#!/usr/bin/env python3
"""
Atelier 2 : Modularisation du Code
Script principal pour l'exécution du pipeline ML
"""

import argparse
import sys
from model_pipeline import (
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

    args = parser.parse_args()

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
            if "X_train" not in locals():
                X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
                    args.train_path, args.test_path
                )

            model = train_model(X_train, y_train, args.model_type)
            save_model(model, args.model_path)

        if args.evaluate:
            print("📈 Évaluation du modèle...")
            model = load_model(args.model_path)

            if "X_test" not in locals() or "y_test" not in locals():
                _, X_test, _, y_test, _, _ = prepare_data(
                    args.train_path, args.test_path
                )

            evaluate_model(model, X_test, y_test, f"Modèle {args.model_type}")

    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)


def run_full_pipeline(args):
    print("\n" + "=" * 50)
    print("ÉTAPE 1: PRÉPARATION DES DONNÉES")
    print("=" * 50)
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
        args.train_path, args.test_path
    )

    print("\n" + "=" * 50)
    print("ÉTAPE 2: ENTRAÎNEMENT DU MODÈLE")
    print("=" * 50)
    model = train_model(X_train, y_train, args.model_type)

    print("\n" + "=" * 50)
    print("ÉTAPE 3: SAUVEGARDE DU MODÈLE")
    print("=" * 50)
    save_model(model, args.model_path)

    print("\n" + "=" * 50)
    print("ÉTAPE 4: ÉVALUATION DU MODÈLE")
    print("=" * 50)
    evaluate_model(model, X_test, y_test, f"Modèle {args.model_type}")

    print("\n🎉 Pipeline exécuté avec succès!")


if __name__ == "__main__":
    main()
