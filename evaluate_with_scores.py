"""
Évaluer le modèle et sauvegarder les scores
"""

from model_pipeline import load_model, evaluate_model, prepare_data
from version_manager import VersionManager

def main():
    try:
        # Charger le modèle
        print("📦 Chargement du modèle...")
        model = load_model()
        
        # Préparer les données
        print("📊 Préparation des données...")
        _, X_test, _, y_test, _, _ = prepare_data()
        
        # Évaluer le modèle - ASSUREZ-VOUS QUE evaluate_model RETOURNE LES MÉTRIQUES
        print("📈 Évaluation du modèle...")
        metrics = evaluate_model(model, X_test, y_test, "Modèle versionné")
        
        # Sauvegarder les scores
        vm = VersionManager()
        vm.save_model_score(metrics)
        
        print(f"✅ Scores sauvegardés pour la version {vm.get_current_version()}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        raise

if __name__ == "__main__":
    main()