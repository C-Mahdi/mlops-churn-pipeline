"""
Module de logging vers Elasticsearch pour MLflow
Gère la connexion et l'envoi des logs vers Elasticsearch
"""
import logging
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
import socket

class ElasticsearchLogger:
    """Classe pour gérer les logs vers Elasticsearch"""
    
    def __init__(self, 
                 hosts=['http://localhost:9200'],
                 index_prefix='mlflow-logs',
                 project_name='churn-prediction'):
        """
        Initialise la connexion à Elasticsearch
        
        Args:
            hosts: Liste des hôtes Elasticsearch
            index_prefix: Préfixe pour les index Elasticsearch
            project_name: Nom du projet pour le tagging
        """
        self.hosts = hosts
        self.index_prefix = index_prefix
        self.project_name = project_name
        self.hostname = socket.gethostname()
        
        # Connexion à Elasticsearch
        try:
            self.es = Elasticsearch(hosts)
            # Tester la connexion
            if self.es.ping():
                print(f"✅ Connexion à Elasticsearch réussie: {hosts}")
                self._create_index_template()
            else:
                print("❌ Impossible de se connecter à Elasticsearch")
                self.es = None
        except Exception as e:
            print(f"❌ Erreur de connexion Elasticsearch: {e}")
            self.es = None
    
    def _create_index_template(self):
        """Crée un template d'index pour structurer les logs"""
        template = {
            "index_patterns": [f"{self.index_prefix}-*"],
            "template": {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                },
                "mappings": {
                    "properties": {
                        "@timestamp": {"type": "date"},
                        "level": {"type": "keyword"},
                        "message": {"type": "text"},
                        "logger_name": {"type": "keyword"},
                        "project_name": {"type": "keyword"},
                        "hostname": {"type": "keyword"},
                        "mlflow_run_id": {"type": "keyword"},
                        "mlflow_experiment_name": {"type": "keyword"},
                        "mlflow_metric_name": {"type": "keyword"},
                        "mlflow_metric_value": {"type": "float"},
                        "mlflow_param_name": {"type": "keyword"},
                        "mlflow_param_value": {"type": "text"},
                        "model_type": {"type": "keyword"},
                        "stage": {"type": "keyword"}
                    }
                }
            }
        }
        
        try:
            self.es.indices.put_index_template(
                name=f"{self.index_prefix}-template",
                body=template
            )
            print(f"✅ Template d'index créé: {self.index_prefix}-template")
        except Exception as e:
            print(f"⚠️  Erreur création template: {e}")
    
    def _get_index_name(self):
        """Génère le nom de l'index avec la date du jour"""
        today = datetime.now().strftime("%Y.%m.%d")
        return f"{self.index_prefix}-{today}"
    
    def log_event(self, level, message, extra_data=None):
        """
        Envoie un log général vers Elasticsearch
        
        Args:
            level: Niveau de log (INFO, WARNING, ERROR, etc.)
            message: Message du log
            extra_data: Données additionnelles (dict)
        """
        if not self.es:
            return
        
        doc = {
            "@timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            "project_name": self.project_name,
            "hostname": self.hostname,
            "logger_name": "mlflow-pipeline"
        }
        
        if extra_data:
            doc.update(extra_data)
        
        try:
            self.es.index(
                index=self._get_index_name(),
                document=doc
            )
        except Exception as e:
            print(f"⚠️  Erreur envoi log: {e}")
    
    def log_mlflow_run_start(self, run_id, experiment_name, model_type):
        """Log le démarrage d'un run MLflow"""
        self.log_event(
            level="INFO",
            message=f"Démarrage du run MLflow: {run_id}",
            extra_data={
                "mlflow_run_id": run_id,
                "mlflow_experiment_name": experiment_name,
                "model_type": model_type,
                "stage": "run_start"
            }
        )
    
    def log_mlflow_run_end(self, run_id, status="SUCCESS"):
        """Log la fin d'un run MLflow"""
        self.log_event(
            level="INFO",
            message=f"Fin du run MLflow: {run_id} - Status: {status}",
            extra_data={
                "mlflow_run_id": run_id,
                "stage": "run_end",
                "status": status
            }
        )
    
    def log_mlflow_params(self, run_id, params):
        """
        Log les paramètres d'un run MLflow
        
        Args:
            run_id: ID du run MLflow
            params: Dictionnaire des paramètres
        """
        if not self.es or not params:
            return
        
        actions = []
        for param_name, param_value in params.items():
            doc = {
                "@timestamp": datetime.utcnow().isoformat(),
                "level": "INFO",
                "message": f"Paramètre MLflow: {param_name} = {param_value}",
                "project_name": self.project_name,
                "mlflow_run_id": run_id,
                "mlflow_param_name": param_name,
                "mlflow_param_value": str(param_value),
                "stage": "parameters"
            }
            actions.append({
                "_index": self._get_index_name(),
                "_source": doc
            })
        
        try:
            bulk(self.es, actions)
            print(f"✅ {len(params)} paramètres envoyés à Elasticsearch")
        except Exception as e:
            print(f"⚠️  Erreur envoi paramètres: {e}")
    
    def log_mlflow_metrics(self, run_id, metrics):
        """
        Log les métriques d'un run MLflow
        
        Args:
            run_id: ID du run MLflow
            metrics: Dictionnaire des métriques
        """
        if not self.es or not metrics:
            return
        
        actions = []
        for metric_name, metric_value in metrics.items():
            doc = {
                "@timestamp": datetime.utcnow().isoformat(),
                "level": "INFO",
                "message": f"Métrique MLflow: {metric_name} = {metric_value}",
                "project_name": self.project_name,
                "mlflow_run_id": run_id,
                "mlflow_metric_name": metric_name,
                "mlflow_metric_value": float(metric_value),
                "stage": "metrics"
            }
            actions.append({
                "_index": self._get_index_name(),
                "_source": doc
            })
        
        try:
            bulk(self.es, actions)
            print(f"✅ {len(metrics)} métriques envoyées à Elasticsearch")
        except Exception as e:
            print(f"⚠️  Erreur envoi métriques: {e}")
    
    def log_data_preparation(self, train_samples, test_samples, n_features):
        """Log les informations de préparation des données"""
        self.log_event(
            level="INFO",
            message="Préparation des données terminée",
            extra_data={
                "stage": "data_preparation",
                "train_samples": train_samples,
                "test_samples": test_samples,
                "n_features": n_features
            }
        )
    
    def log_model_training(self, model_type, duration=None):
        """Log les informations d'entraînement du modèle"""
        message = f"Entraînement du modèle {model_type}"
        if duration:
            message += f" - Durée: {duration:.2f}s"
        
        extra_data = {
            "stage": "model_training",
            "model_type": model_type
        }
        if duration:
            extra_data["training_duration"] = duration
        
        self.log_event(level="INFO", message=message, extra_data=extra_data)
    
    def log_model_evaluation(self, model_type, metrics):
        """Log les résultats d'évaluation du modèle"""
        self.log_event(
            level="INFO",
            message=f"Évaluation du modèle {model_type} terminée",
            extra_data={
                "stage": "model_evaluation",
                "model_type": model_type,
                **{f"metric_{k}": v for k, v in metrics.items()}
            }
        )
    
    def log_error(self, error_message, exception=None):
        """Log une erreur"""
        extra_data = {"stage": "error"}
        if exception:
            extra_data["exception_type"] = type(exception).__name__
            extra_data["exception_message"] = str(exception)
        
        self.log_event(
            level="ERROR",
            message=error_message,
            extra_data=extra_data
        )
    
    def search_logs(self, query=None, size=100):
        """
        Recherche dans les logs Elasticsearch
        
        Args:
            query: Requête Elasticsearch (dict)
            size: Nombre de résultats à retourner
        
        Returns:
            Liste des documents trouvés
        """
        if not self.es:
            return []
        
        try:
            if query is None:
                query = {"match_all": {}}
            
            result = self.es.search(
                index=f"{self.index_prefix}-*",
                query=query,
                size=size,
                sort=[{"@timestamp": {"order": "desc"}}]
            )
            
            return [hit["_source"] for hit in result["hits"]["hits"]]
        except Exception as e:
            print(f"⚠️  Erreur recherche logs: {e}")
            return []
    
    def get_run_logs(self, run_id):
        """Récupère tous les logs d'un run spécifique"""
        query = {
            "term": {"mlflow_run_id": run_id}
        }
        return self.search_logs(query=query)
    
    def close(self):
        """Ferme la connexion Elasticsearch"""
        if self.es:
            self.es.close()
            print("✅ Connexion Elasticsearch fermée")


# Instance globale pour faciliter l'utilisation
_global_es_logger = None

def get_elasticsearch_logger(hosts=['http://localhost:9200'], 
                             index_prefix='mlflow-logs',
                             project_name='churn-prediction'):
    """
    Récupère ou crée l'instance globale du logger Elasticsearch
    
    Returns:
        ElasticsearchLogger instance
    """
    global _global_es_logger
    if _global_es_logger is None:
        _global_es_logger = ElasticsearchLogger(
            hosts=hosts,
            index_prefix=index_prefix,
            project_name=project_name
        )
    return _global_es_logger