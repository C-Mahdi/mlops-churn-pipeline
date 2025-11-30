"""
Gestionnaire de versions pour les modèles ML
"""

import os
import json
import joblib
from datetime import datetime
import shutil

class VersionManager:
    def __init__(self, base_path="models"):
        self.base_path = base_path
        self.versions_path = os.path.join(base_path, "versions")
        self.version_file = os.path.join(base_path, "current_version.txt")
        self.metadata_file = os.path.join(base_path, "model_metadata.json")
        
        # Créer les dossiers si nécessaire
        os.makedirs(self.versions_path, exist_ok=True)
    
    def get_current_version(self):
        """Obtenir la version actuelle"""
        try:
            with open(self.version_file, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            return "1.0"
    
    def get_next_version(self):
        """Obtenir la prochaine version"""
        current = self.get_current_version()
        try:
            major, minor = map(int, current.split('.'))
            minor += 1
            return f"{major}.{minor}"
        except:
            return "1.1"
    
    def save_current_version(self, version=None):
        """Sauvegarder la version actuelle"""
        if version is None:
            version = self.get_next_version()
        
        with open(self.version_file, 'w') as f:
            f.write(version)
        
        # Mettre à jour les métadonnées
        self._update_metadata(version)
        
        return version
    
    def _update_metadata(self, version):
        """Mettre à jour les métadonnées du modèle"""
        metadata = {
            'version': version,
            'last_trained': datetime.now().isoformat()
        }
        
        # Charger les métadonnées existantes
        try:
            with open(self.metadata_file, 'r') as f:
                existing_metadata = json.load(f)
        except FileNotFoundError:
            existing_metadata = {}
        
        # Mettre à jour l'historique des versions
        if 'version_history' not in existing_metadata:
            existing_metadata['version_history'] = []
        
        existing_metadata['version_history'].append({
            'version': version,
            'timestamp': metadata['last_trained']
        })
        
        # Fusionner les métadonnées
        existing_metadata.update(metadata)
        
        # Sauvegarder
        with open(self.metadata_file, 'w') as f:
            json.dump(existing_metadata, f, indent=2)
    
    def save_model_score(self, scores):
        """Sauvegarder les scores d'évaluation pour la version actuelle"""
        version = self.get_current_version()
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = {}
        
        if 'version_scores' not in metadata:
            metadata['version_scores'] = {}
        
        metadata['version_scores'][version] = {
            'scores': scores,
            'evaluation_date': datetime.now().isoformat()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Scores sauvegardés pour la version {version}")
    
    def get_model_scores(self):
        """Obtenir tous les scores par version"""
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            return metadata.get('version_scores', {})
        except FileNotFoundError:
            return {}
    
    def create_version_snapshot(self, version=None):
        """Créer un snapshot de la version"""
        if version is None:
            version = self.get_current_version()
        
        version_path = os.path.join(self.versions_path, f"v{version}")
        os.makedirs(version_path, exist_ok=True)
        
        # Copier les fichiers du modèle
        model_files = [
            'churn_model.joblib',
            'scaler.joblib',
            'encoder_state.joblib', 
            'encoder_area.joblib'
        ]
        
        for file in model_files:
            src = os.path.join(self.base_path, file)
            dst = os.path.join(version_path, file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        # Copier les métadonnées
        if os.path.exists(self.metadata_file):
            shutil.copy2(self.metadata_file, version_path)
        
        print(f"✅ Snapshot version {version} créé")
    
    def list_versions(self):
        """Lister toutes les versions disponibles"""
        versions = []
        if os.path.exists(self.versions_path):
            for item in os.listdir(self.versions_path):  # Correction: os.listdir
                if item.startswith('v'):
                    versions.append(item[1:])
        return sorted(versions)
    
    def show_scores(self):
        """Afficher les scores de toutes les versions"""
        scores_data = self.get_model_scores()
        
        if not scores_data:
            print("📭 Aucun score enregistré")
            print("💡 Utilisez 'make evaluate-version' pour enregistrer les scores")
            return
        
        print("📈 SCORES PAR VERSION")
        print("=" * 50)
        
        for version in sorted(scores_data.keys()):
            version_data = scores_data[version]
            scores = version_data['scores']
            
            print(f"\n🎯 Version {version}:")
            for metric, value in scores.items():
                print(f"   {metric}: {value:.4f}")

if __name__ == "__main__":
    vm = VersionManager()
    print(f"Version actuelle: {vm.get_current_version()}")
    print(f"Prochaine version: {vm.get_next_version()}")
    print(f"Versions disponibles: {vm.list_versions()}")