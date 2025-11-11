"""
Pipeline MLOps com MLflow para versionamento e tracking de modelos.
"""

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Optional
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowPipeline:
    """Pipeline MLOps usando MLflow."""
    
    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: str = "workwell-ai"):
        """
        Inicializa pipeline MLflow.
        
        Args:
            tracking_uri: URI do servidor MLflow
            experiment_name: Nome do experimento
        """
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Criar ou obter experimento
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Experimento '{experiment_name}' criado")
        except:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            logger.info(f"Usando experimento existente '{experiment_name}'")
        
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
    
    def start_run(self, run_name: Optional[str] = None):
        """Inicia uma nova run do MLflow."""
        return mlflow.start_run(run_name=run_name)
    
    def log_model_params(self, params: Dict):
        """Registra parâmetros do modelo."""
        mlflow.log_params(params)
        logger.info(f"Parâmetros registrados: {params}")
    
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """Registra métricas."""
        mlflow.log_metrics(metrics, step=step)
        logger.info(f"Métricas registradas: {metrics}")
    
    def log_artifacts(self, artifact_path: str, artifact_dir: str):
        """Registra artefatos (arquivos)."""
        mlflow.log_artifacts(artifact_dir, artifact_path)
        logger.info(f"Artefatos registrados de {artifact_dir}")
    
    def log_pytorch_model(self, model, artifact_path: str = "model"):
        """Registra modelo PyTorch."""
        mlflow.pytorch.log_model(model, artifact_path)
        logger.info(f"Modelo PyTorch registrado em {artifact_path}")
    
    def log_sklearn_model(self, model, artifact_path: str = "model"):
        """Registra modelo scikit-learn."""
        mlflow.sklearn.log_model(model, artifact_path)
        logger.info(f"Modelo scikit-learn registrado em {artifact_path}")
    
    def register_model(self, model_uri: str, model_name: str):
        """Registra modelo no Model Registry."""
        mlflow.register_model(model_uri, model_name)
        logger.info(f"Modelo '{model_name}' registrado")
    
    def load_model(self, model_name: str, version: Optional[int] = None, stage: Optional[str] = None):
        """
        Carrega modelo do Model Registry.
        
        Args:
            model_name: Nome do modelo
            version: Versão específica (opcional)
            stage: Stage (Production, Staging, etc.)
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
        elif stage:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            model_uri = f"models:/{model_name}/latest"
        
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Modelo '{model_name}' carregado de {model_uri}")
        return model
    
    def validate_model(self, model, validation_data, metrics_func) -> Dict:
        """
        Valida modelo antes de produção.
        
        Args:
            model: Modelo a validar
            validation_data: Dados de validação
            metrics_func: Função para calcular métricas
            
        Returns:
            Dicionário com métricas de validação
        """
        metrics = metrics_func(model, validation_data)
        
        # Verificar thresholds
        min_accuracy = 0.7
        if metrics.get('accuracy', 0) < min_accuracy:
            raise ValueError(f"Accuracy abaixo do threshold: {metrics['accuracy']}")
        
        logger.info(f"Validação passou: {metrics}")
        return metrics
    
    def promote_to_production(self, model_name: str, version: int):
        """Promove modelo para produção."""
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        logger.info(f"Modelo '{model_name}' versão {version} promovido para Production")


def train_and_log_burnout_model(
    model,
    train_data,
    val_data,
    params: Dict,
    metrics: Dict
):
    """Treina e registra modelo de burnout no MLflow."""
    pipeline = MLflowPipeline()
    
    with pipeline.start_run(run_name=f"burnout_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parâmetros
        pipeline.log_model_params(params)
        
        # Log métricas
        pipeline.log_metrics(metrics)
        
        # Log modelo
        pipeline.log_pytorch_model(model, "burnout_model")
        
        # Registrar modelo
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/burnout_model"
        pipeline.register_model(model_uri, "burnout-predictor")
        
        logger.info("Modelo de burnout treinado e registrado")


if __name__ == "__main__":
    # Exemplo de uso
    pipeline = MLflowPipeline()
    
    # Iniciar run
    with pipeline.start_run():
        # Log parâmetros
        pipeline.log_model_params({
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        })
        
        # Log métricas
        pipeline.log_metrics({
            "train_loss": 0.5,
            "val_loss": 0.6,
            "accuracy": 0.85
        })

