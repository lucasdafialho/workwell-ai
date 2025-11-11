"""
Sistema de Monitoramento e Visualizações para métricas de IA.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly não disponível. Visualizações serão limitadas.")


class AIMonitoring:
    """
    Sistema de monitoramento para métricas de ML em produção.
    """
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.predictions_history = []
        self.model_performance = {}
    
    def log_prediction(
        self,
        model_name: str,
        prediction: Dict,
        actual: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Registra predição para monitoramento.
        
        Args:
            model_name: Nome do modelo
            prediction: Dicionário com predição
            actual: Valor real (se disponível)
            timestamp: Timestamp da predição
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        log_entry = {
            'model_name': model_name,
            'timestamp': timestamp,
            'prediction': prediction,
            'actual': actual,
            'error': abs(prediction.get('value', 0) - actual) if actual else None
        }
        
        self.predictions_history.append(log_entry)
        self.metrics_history[model_name].append(log_entry)
    
    def calculate_metrics(
        self,
        model_name: str,
        window_days: int = 7
    ) -> Dict:
        """
        Calcula métricas de performance do modelo.
        
        Args:
            model_name: Nome do modelo
            window_days: Janela de tempo para análise
            
        Returns:
            Dicionário com métricas
        """
        cutoff_date = datetime.now() - timedelta(days=window_days)
        
        recent_predictions = [
            p for p in self.metrics_history[model_name]
            if p['timestamp'] >= cutoff_date and p['actual'] is not None
        ]
        
        if not recent_predictions:
            return {}
        
        predictions = [p['prediction'].get('value', 0) for p in recent_predictions]
        actuals = [p['actual'] for p in recent_predictions]
        errors = [p['error'] for p in recent_predictions if p['error'] is not None]
        
        metrics = {
            'n_predictions': len(recent_predictions),
            'mae': np.mean(errors) if errors else None,
            'rmse': np.sqrt(np.mean([e**2 for e in errors])) if errors else None,
            'accuracy': self._calculate_accuracy(predictions, actuals),
            'precision': self._calculate_precision(predictions, actuals),
            'recall': self._calculate_recall(predictions, actuals),
            'f1_score': self._calculate_f1(predictions, actuals)
        }
        
        return metrics
    
    def detect_data_drift(
        self,
        model_name: str,
        current_data: np.ndarray,
        reference_data: np.ndarray
    ) -> Dict:
        """
        Detecta data drift comparando distribuições.
        
        Args:
            model_name: Nome do modelo
            current_data: Dados atuais
            reference_data: Dados de referência
            
        Returns:
            Dicionário com detecção de drift
        """
        from scipy import stats
        
        drift_detected = False
        drift_scores = {}
        
        # Kolmogorov-Smirnov test para cada feature
        if current_data.ndim == 1:
            current_data = current_data.reshape(-1, 1)
            reference_data = reference_data.reshape(-1, 1)
        
        for i in range(current_data.shape[1]):
            ks_statistic, p_value = stats.ks_2samp(
                reference_data[:, i],
                current_data[:, i]
            )
            
            drift_scores[f'feature_{i}'] = {
                'ks_statistic': float(ks_statistic),
                'p_value': float(p_value),
                'drift_detected': p_value < 0.05
            }
            
            if p_value < 0.05:
                drift_detected = True
        
        return {
            'drift_detected': drift_detected,
            'drift_scores': drift_scores,
            'timestamp': datetime.now()
        }
    
    def _calculate_accuracy(self, predictions: List, actuals: List) -> float:
        """Calcula accuracy."""
        if len(predictions) == 0:
            return 0.0
        
        correct = sum(1 for p, a in zip(predictions, actuals) if abs(p - a) < 0.1)
        return correct / len(predictions)
    
    def _calculate_precision(self, predictions: List, actuals: List) -> float:
        """Calcula precision."""
        # Simplificado - em produção, usar métricas apropriadas
        return self._calculate_accuracy(predictions, actuals)
    
    def _calculate_recall(self, predictions: List, actuals: List) -> float:
        """Calcula recall."""
        return self._calculate_accuracy(predictions, actuals)
    
    def _calculate_f1(self, predictions: List, actuals: List) -> float:
        """Calcula F1 score."""
        precision = self._calculate_precision(predictions, actuals)
        recall = self._calculate_recall(predictions, actuals)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)


class VisualizationDashboard:
    """
    Cria visualizações interativas para monitoramento de IA.
    """
    
    @staticmethod
    def create_burnout_risk_heatmap(
        data: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> go.Figure:
        """
        Cria heatmap de padrões de stress por departamento.
        
        Args:
            data: DataFrame com dados
            output_path: Caminho para salvar (opcional)
            
        Returns:
            Figura Plotly
        """
        # Preparar dados
        pivot_data = data.pivot_table(
            index='departamento',
            columns='data_checkin',
            values='nivel_stress',
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlGn_r',
            colorbar=dict(title="Nível de Stress")
        ))
        
        fig.update_layout(
            title="Heatmap de Stress por Departamento",
            xaxis_title="Data",
            yaxis_title="Departamento"
        )
        
        if output_path:
            fig.write_html(output_path)
        
        return fig
    
    @staticmethod
    def create_trend_forecast_plot(
        historical: pd.DataFrame,
        forecast: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> go.Figure:
        """
        Cria gráfico de tendência com previsões.
        
        Args:
            historical: Dados históricos
            forecast: Dados de previsão
            output_path: Caminho para salvar (opcional)
            
        Returns:
            Figura Plotly
        """
        fig = go.Figure()
        
        # Dados históricos
        fig.add_trace(go.Scatter(
            x=historical['date'],
            y=historical['value'],
            mode='lines',
            name='Histórico',
            line=dict(color='blue')
        ))
        
        # Previsão
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['predicted'],
            mode='lines',
            name='Previsão',
            line=dict(color='green', dash='dash')
        ))
        
        # Intervalo de confiança
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['upper'],
            mode='lines',
            name='Limite Superior',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['lower'],
            mode='lines',
            name='Intervalo de Confiança',
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(width=0)
        ))
        
        fig.update_layout(
            title="Previsão de Bem-estar",
            xaxis_title="Data",
            yaxis_title="Score de Bem-estar"
        )
        
        if output_path:
            fig.write_html(output_path)
        
        return fig
    
    @staticmethod
    def create_correlation_matrix(
        data: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> go.Figure:
        """
        Cria matriz de correlação entre métricas.
        
        Args:
            data: DataFrame com métricas
            output_path: Caminho para salvar (opcional)
            
        Returns:
            Figura Plotly
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Correlação")
        ))
        
        fig.update_layout(
            title="Matriz de Correlação entre Métricas",
            width=800,
            height=600
        )
        
        if output_path:
            fig.write_html(output_path)
        
        return fig
    
    @staticmethod
    def create_model_performance_dashboard(
        metrics: Dict[str, Dict],
        output_path: Optional[str] = None
    ) -> go.Figure:
        """
        Cria dashboard de performance de modelos.
        
        Args:
            metrics: Dicionário {model_name: {metric_name: value}}
            output_path: Caminho para salvar (opcional)
            
        Returns:
            Figura Plotly com subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1 Score'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        model_names = list(metrics.keys())
        
        # Accuracy
        accuracies = [metrics[m].get('accuracy', 0) for m in model_names]
        fig.add_trace(go.Bar(x=model_names, y=accuracies, name='Accuracy'), row=1, col=1)
        
        # Precision
        precisions = [metrics[m].get('precision', 0) for m in model_names]
        fig.add_trace(go.Bar(x=model_names, y=precisions, name='Precision'), row=1, col=2)
        
        # Recall
        recalls = [metrics[m].get('recall', 0) for m in model_names]
        fig.add_trace(go.Bar(x=model_names, y=recalls, name='Recall'), row=2, col=1)
        
        # F1
        f1_scores = [metrics[m].get('f1_score', 0) for m in model_names]
        fig.add_trace(go.Bar(x=model_names, y=f1_scores, name='F1 Score'), row=2, col=2)
        
        fig.update_layout(
            title="Dashboard de Performance de Modelos",
            height=600,
            showlegend=False
        )
        
        if output_path:
            fig.write_html(output_path)
        
        return fig


if __name__ == "__main__":
    # Exemplo de uso
    monitoring = AIMonitoring()
    
    # Log predições
    monitoring.log_prediction(
        model_name="burnout_predictor",
        prediction={"value": 0.75, "class": "alto"},
        actual=0.70
    )
    
    # Calcular métricas
    metrics = monitoring.calculate_metrics("burnout_predictor")
    print(f"Métricas: {metrics}")

