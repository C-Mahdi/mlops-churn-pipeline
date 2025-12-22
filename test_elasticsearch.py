#!/usr/bin/env python3
"""
Script de test de la connexion du Elasticsearch
"""
import sys
from elasticsearch import Elasticsearch
from datetime import datetime

def test_elasticsearch_connection(host='http://localhost:9200'):
    """Test la connexion à Elasticsearch"""
    
    print("\n" + "=" * 70)
    print("🧪 TEST DE CONNEXION ELASTICSEARCH")
    print("=" * 70)
    
    print(f"\n1️⃣  Test de connexion à {host}...")
    
    try:
        es = Elasticsearch([host])
        
        if es.ping():
            print("   ✅ Connexion réussie!")
        else:
            print("   ❌ Impossible de se connecter")
            return False
    except Exception as e:
        print(f"   ❌ Erreur de connexion: {e}")
        print("\n💡 Assurez-vous que:")
        print("   - Docker est démarré")
        print("   - Les conteneurs sont en cours d'exécution (docker ps)")
        print("   - Elasticsearch est accessible sur le port 9200")
        return False
    
    # Test 2: Informations sur le cluster
    print("\n2️⃣  Récupération des informations du cluster...")
    try:
        info = es.info()
        print(f"   ✅ Cluster: {info['cluster_name']}")
        print(f"   ✅ Version: {info['version']['number']}")
    except Exception as e:
        print(f"   ⚠️  Erreur: {e}")
    
    # Test 3: Création d'un index de test
    print("\n3️⃣  Test de création d'index...")
    test_index = "test-index"
    
    try:
        # Supprimer l'index s'il existe
        if es.indices.exists(index=test_index):
            es.indices.delete(index=test_index)
            print(f"   🗑️  Index existant supprimé")
        
        # Créer l'index
        es.indices.create(index=test_index)
        print(f"   ✅ Index '{test_index}' créé")
    except Exception as e:
        print(f"   ⚠️  Erreur création index: {e}")
    
    # Test 4: Insertion d'un document de test
    print("\n4️⃣  Test d'insertion de document...")
    
    try:
        doc = {
            "@timestamp": datetime.utcnow().isoformat(),
            "level": "INFO",
            "message": "Test document from test script",
            "test": True
        }
        
        result = es.index(index=test_index, document=doc)
        print(f"   ✅ Document inséré avec ID: {result['_id']}")
    except Exception as e:
        print(f"   ⚠️  Erreur insertion: {e}")
    
    # Test 5: Recherche du document
    print("\n5️⃣  Test de recherche...")
    
    try:
        # Rafraîchir l'index pour que le document soit cherchable
        es.indices.refresh(index=test_index)
        
        result = es.search(
            index=test_index,
            query={"match": {"test": True}}
        )
        
        count = result['hits']['total']['value']
        print(f"   ✅ {count} document(s) trouvé(s)")
        
        if count > 0:
            print(f"   📄 Message: {result['hits']['hits'][0]['_source']['message']}")
    except Exception as e:
        print(f"   ⚠️  Erreur recherche: {e}")
    
    # Test 6: Nettoyage
    print("\n6️⃣  Nettoyage...")
    
    try:
        es.indices.delete(index=test_index)
        print(f"   ✅ Index de test supprimé")
    except Exception as e:
        print(f"   ⚠️  Erreur nettoyage: {e}")
    
    # Test 7: Liste des index existants
    print("\n7️⃣  Liste des index existants...")
    
    try:
        indices = es.indices.get_alias(index="*")
        if indices:
            print(f"   📋 {len(indices)} index trouvé(s):")
            for idx in list(indices.keys())[:10]:  # Afficher max 10
                print(f"      - {idx}")
            if len(indices) > 10:
                print(f"      ... et {len(indices) - 10} autre(s)")
        else:
            print("   ℹ️  Aucun index existant")
    except Exception as e:
        print(f"   ⚠️  Erreur: {e}")
    
    es.close()
    
    print("\n" + "=" * 70)
    print("✅ TOUS LES TESTS SONT PASSÉS")
    print("=" * 70)
    print("\n💡 Prochaines étapes:")
    print("   1. Exécutez votre pipeline: python main.py --full-pipeline")
    print("   2. Les logs seront envoyés à Elasticsearch")
    print("   3. Visualisez dans Kibana: http://localhost:5601")
    print("\n" + "=" * 70 + "\n")
    
    return True

def test_elasticsearch_logger():
    """Test le module elasticsearch_logger"""
    
    print("\n" + "=" * 70)
    print("🧪 TEST DU MODULE ELASTICSEARCH_LOGGER")
    print("=" * 70)
    
    try:
        from elasticsearch_logger import get_elasticsearch_logger
        
        print("\n1️⃣  Import du module...")
        print("   ✅ Module importé avec succès")
        
        print("\n2️⃣  Création de l'instance logger...")
        es_logger = get_elasticsearch_logger()
        
        if es_logger.es is None:
            print("   ❌ Logger non initialisé - Vérifiez la connexion ES")
            return False
        
        print("   ✅ Logger créé avec succès")
        
        print("\n3️⃣  Test d'envoi de log...")
        es_logger.log_event(
            "INFO",
            "Test log from test script",
            {"test": True, "timestamp": datetime.utcnow().isoformat()}
        )
        print("   ✅ Log envoyé")
        
        print("\n4️⃣  Test d'envoi de métriques MLflow...")
        test_metrics = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88
        }
        es_logger.log_mlflow_metrics("test-run-id-123", test_metrics)
        print("   ✅ Métriques envoyées")
        
        print("\n5️⃣  Test de recherche de logs...")
        logs = es_logger.search_logs(size=5)
        print(f"   ✅ {len(logs)} log(s) récupéré(s)")
        
        if logs:
            print(f"   📄 Dernier log: {logs[0].get('message', 'N/A')}")
        
        es_logger.close()
        
        print("\n" + "=" * 70)
        print("✅ MODULE ELASTICSEARCH_LOGGER FONCTIONNE CORRECTEMENT")
        print("=" * 70 + "\n")
        
        return True
        
    except ImportError as e:
        print(f"\n   ❌ Erreur d'import: {e}")
        print("\n💡 Assurez-vous que le fichier elasticsearch_logger.py existe")
        return False
    except Exception as e:
        print(f"\n   ❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    print("\n🚀 TESTS D'INTÉGRATION ELASTICSEARCH\n")
    
    # Test 1: Connexion Elasticsearch
    success1 = test_elasticsearch_connection()
    
    if success1:
        # Test 2: Module elasticsearch_logger
        success2 = test_elasticsearch_logger()
        
        if success1 and success2:
            print("\n" + "=" * 70)
            print("🎉 TOUS LES TESTS SONT PASSÉS AVEC SUCCÈS!")
            print("=" * 70)
            print("\n✅ Votre environnement ELK est prêt à être utilisé!")
            print("\n📚 Consultez le guide: Guide_Integration_ELK.md")
            print("=" * 70 + "\n")
            sys.exit(0)
    
    print("\n" + "=" * 70)
    print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
    print("=" * 70)
    print("\n💡 Vérifiez les erreurs ci-dessus et corrigez-les avant de continuer")
    print("=" * 70 + "\n")
    sys.exit(1)