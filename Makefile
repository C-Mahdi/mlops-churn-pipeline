# Configuration
PYTHON = python3
ENV_NAME = mlops_env
REQUIREMENTS = requirements.txt

# Fichiers sources pour le CI/CD
PYTHON_FILES = model_pipeline.py main.py version_manager.py evaluate_with_scores.py
DATA_FILES = data/raw/churn-bigml-80.csv data/raw/churn-bigml-20.csv

.PHONY: setup install code-quality data train train-version evaluate evaluate-version scores list-versions clean help all ci-cd

.DEFAULT_GOAL := help

## 🚀 Installation des dépendances
setup:
	@echo "=== Création de l'environnement virtuel ==="
	$(PYTHON) -m venv $(ENV_NAME)
	@echo "✅ Environnement virtuel créé"
	@echo "=== Installation des dépendances depuis requirements.txt ==="
	@./$(ENV_NAME)/bin/pip install -r $(REQUIREMENTS)
	@echo "✅ Dépendances installées"

## 🔍 Vérification du code
code-quality:
	@echo "=== Installation des outils de qualité de code ==="
	@./$(ENV_NAME)/bin/pip install black flake8 bandit
	@echo "=== Formatage automatique du code ==="
	@./$(ENV_NAME)/bin/black $(PYTHON_FILES)
	@echo "✅ Code formaté avec Black"
	@echo "=== Vérification de la qualité du code ==="
	@./$(ENV_NAME)/bin/flake8 $(PYTHON_FILES) --max-line-length=100 --ignore=E203,W503
	@echo "✅ Qualité du code vérifiée avec Flake8"
	@echo "=== Analyse de sécurité du code ==="
	@./$(ENV_NAME)/bin/bandit -r . -f html -o reports/security_report.html
	@echo "✅ Sécurité analysée avec Bandit"

## 📊 Préparation des données
data:
	@echo "=== Préparation des données ==="
	@./$(ENV_NAME)/bin/python main.py --prepare
	@echo "✅ Données préparées"

## 🤖 Entraînement du modèle
train:
	@echo "=== Entraînement du modèle ==="
	@./$(ENV_NAME)/bin/python main.py --train
	@echo "✅ Modèle entraîné"

## 🏷️ Entraînement avec versionning
train-version:
	@echo "=== Entraînement avec versionning ==="
	@./$(ENV_NAME)/bin/python -c "from version_manager import VersionManager; vm = VersionManager(); version = vm.get_next_version(); print(f'🎯 Utilisation de la version: {version}')"
	@./$(ENV_NAME)/bin/python main.py --train
	@./$(ENV_NAME)/bin/python -c "from version_manager import VersionManager; vm = VersionManager(); vm.save_current_version(); vm.create_version_snapshot(); print(f'✅ Modèle version {vm.get_current_version()} sauvegardé et versionné')"

## 📈 Évaluation du modèle
evaluate:
	@echo "=== Évaluation du modèle ==="
	@./$(ENV_NAME)/bin/python main.py --evaluate
	@echo "✅ Modèle évalué"

## 🎯 Évaluation avec sauvegarde des scores
evaluate-version:
	@echo "=== Évaluation avec scores ==="
	@./$(ENV_NAME)/bin/python evaluate_with_scores.py
	@echo "✅ Scores sauvegardés"

## 📊 Afficher les scores
scores:
	@echo "=== Scores par version ==="
	@./$(ENV_NAME)/bin/python -c "from version_manager import VersionManager; vm = VersionManager(); vm.show_scores()"

## 📋 Lister les versions
list-versions:
	@echo "=== Versions disponibles ==="
	@./$(ENV_NAME)/bin/python -c "from version_manager import VersionManager; vm = VersionManager(); print(f'Version actuelle: {vm.get_current_version()}'); print('Versions:'); [print(f'  - {v}') for v in vm.list_versions()]"

## 🧪 Tests
test:
	@echo "=== Tests ==="
	@mkdir -p reports
	@./$(ENV_NAME)/bin/python -c "import pandas as pd; import sklearn; print('✅ Bibliothèques OK')" > reports/test_results.txt
	@./$(ENV_NAME)/bin/python -c "from model_pipeline import prepare_data, train_model; print('✅ Modules OK')" >> reports/test_results.txt
	@./$(ENV_NAME)/bin/python -c "from version_manager import VersionManager; print('✅ Version manager OK')" >> reports/test_results.txt
	@echo "✅ Tests terminés - voir reports/test_results.txt"

## 🔄 CI/CD - Pipeline automatique
ci-cd: $(PYTHON_FILES) $(DATA_FILES)
	@echo "🚀 DÉMARRAGE DU PIPELINE CI/CD"
	@echo "📁 Fichiers modifiés détectés: $?"
	@date > reports/last_ci_cd.txt
	@echo "=== Étape 1: Qualité du code ==="
	@$(MAKE) code-quality
	@echo "=== Étape 2: Préparation des données ==="
	@$(MAKE) data
	@echo "=== Étape 3: Entraînement avec versionning ==="
	@$(MAKE) train-version
	@echo "=== Étape 4: Évaluation avec scores ==="
	@$(MAKE) evaluate-version
	@echo "=== Étape 5: Tests ==="
	@$(MAKE) test
	@echo "🎉 PIPELINE CI/CD TERMINÉ AVEC SUCCÈS"
	@echo "📊 Rapport généré: reports/last_ci_cd.txt"

## 🎯 Pipeline complet avec versionning
## 🎯 Pipeline complet avec versionning
all: setup data train-version evaluate-version test
	@echo "=== TOUTES LES ÉTAPES TERMINÉES ==="
	@echo "✅ Projet entièrement configuré et prêt"

	
## 🧹 Nettoyage
clean:
	@echo "=== Nettoyage ==="
	rm -rf models/*
	rm -rf reports/*
	rm -rf __pycache__
	rm -f *.pyc
	@echo "✅ Fichiers générés nettoyés"

## 📋 Aide
## 📋 Aide
help:
	@echo "=== MAKEFILE - PRÉDICTION DE CHURN ==="
	@echo "Commandes disponibles:"
	@echo "  setup           - Créer env virtuel + installer dépendances"
	@echo "  code-quality    - Formatage, qualité, sécurité du code"
	@echo "  data            - Préparer les données"
	@echo "  train           - Entraîner le modèle (simple)"
	@echo "  train-version   - Entraîner avec versionning automatique"
	@echo "  evaluate        - Évaluer le modèle"
	@echo "  evaluate-version- Évaluer et sauvegarder les scores"
	@echo "  scores          - Afficher les scores par version"
	@echo "  list-versions   - Lister les versions de modèles"
	@echo "  test            - Tests unitaires"
	@echo "  api             - Démarrer l'API FastAPI"
	@echo "  api-prod        - API en mode production"
	@echo "  test-api        - Tester l'API"
	@echo "  ci-cd           - Pipeline CI/CD complet (automatique)"
	@echo "  all             - Tout exécuter (setup + pipeline)"
	@echo "  clean           - Nettoyer les fichiers générés"
	@echo "  help            - Afficher cette aide"

## 🚀 API REST
api:
	@echo "=== Démarrage de l'API FastAPI ==="
	@./$(ENV_NAME)/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --reload

## 📡 API en production
api-prod:
	@echo "=== Démarrage de l'API en mode production ==="
	@./$(ENV_NAME)/bin/uvicorn app:app --host 0.0.0.0 --port 8000

## 🧪 Test de l'API
test-api:
	@echo "=== Test de l'API ==="
	@./$(ENV_NAME)/bin/python -c "\
	import requests; \
	import time; \
	\
	# Attendre que l'API démarre \
	print('Démarrage du test API...'); \
	time.sleep(2); \
	\
	# Test de santé \
	try: \
	    response = requests.get('http://localhost:8000/health'); \
	    print(f'✅ Health check: {response.status_code}'); \
	    print(f'   Réponse: {response.json()}'); \
	except Exception as e: \
	    print(f'❌ Health check échoué: {e}'); \
	\
	# Test des infos modèle \
	try: \
	    response = requests.get('http://localhost:8000/model-info'); \
	    print(f'✅ Model info: {response.status_code}'); \
	    print(f'   Version: {response.json().get(\"version\", \"inconnue\")}'); \
	except Exception as e: \
	    print(f'❌ Model info échoué: {e}'); \
	"