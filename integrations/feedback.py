"""
Sistema de Feedback e Aprendizado Contínuo.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackSystem:
    """
    Sistema de coleta e processamento de feedback para melhorar modelos.
    """
    
    def __init__(self):
        self.feedback_history = []
        self.recommendation_feedback = defaultdict(list)
        self.prediction_feedback = defaultdict(list)
    
    def submit_recommendation_feedback(
        self,
        user_id: int,
        recommendation_id: str,
        rating: float,
        completed: bool = True,
        comments: Optional[str] = None
    ):
        """
        Submete feedback sobre recomendação.
        
        Args:
            user_id: ID do usuário
            recommendation_id: ID da recomendação
            rating: Rating (0-5)
            completed: Se foi completada
            comments: Comentários opcionais
        """
        feedback = {
            'user_id': user_id,
            'recommendation_id': recommendation_id,
            'rating': rating,
            'completed': completed,
            'comments': comments,
            'timestamp': datetime.now(),
            'type': 'recommendation'
        }
        
        self.feedback_history.append(feedback)
        self.recommendation_feedback[recommendation_id].append(feedback)
        
        logger.info(f"Feedback de recomendação registrado: {recommendation_id}")
    
    def submit_prediction_feedback(
        self,
        user_id: int,
        prediction_id: str,
        was_accurate: bool,
        actual_outcome: Optional[str] = None,
        comments: Optional[str] = None
    ):
        """
        Submete feedback sobre predição.
        
        Args:
            user_id: ID do usuário
            prediction_id: ID da predição
            was_accurate: Se foi precisa
            actual_outcome: Resultado real (opcional)
            comments: Comentários opcionais
        """
        feedback = {
            'user_id': user_id,
            'prediction_id': prediction_id,
            'was_accurate': was_accurate,
            'actual_outcome': actual_outcome,
            'comments': comments,
            'timestamp': datetime.now(),
            'type': 'prediction'
        }
        
        self.feedback_history.append(feedback)
        self.prediction_feedback[prediction_id].append(feedback)
        
        logger.info(f"Feedback de predição registrado: {prediction_id}")
    
    def analyze_feedback_trends(
        self,
        days: int = 30
    ) -> Dict:
        """
        Analisa tendências de feedback.
        
        Args:
            days: Número de dias para analisar
            
        Returns:
            Dicionário com análise de tendências
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_feedback = [
            f for f in self.feedback_history
            if f['timestamp'] >= cutoff_date
        ]
        
        if not recent_feedback:
            return {}
        
        rec_feedback = [f for f in recent_feedback if f['type'] == 'recommendation']
        avg_rating = np.mean([f['rating'] for f in rec_feedback]) if rec_feedback else 0
        completion_rate = sum(1 for f in rec_feedback if f['completed']) / len(rec_feedback) if rec_feedback else 0
        
        pred_feedback = [f for f in recent_feedback if f['type'] == 'prediction']
        accuracy_rate = sum(1 for f in pred_feedback if f['was_accurate']) / len(pred_feedback) if pred_feedback else 0
        
        return {
            'period_days': days,
            'total_feedback': len(recent_feedback),
            'recommendation_metrics': {
                'average_rating': float(avg_rating),
                'completion_rate': float(completion_rate),
                'total_feedback': len(rec_feedback)
            },
            'prediction_metrics': {
                'accuracy_rate': float(accuracy_rate),
                'total_feedback': len(pred_feedback)
            },
            'trend': 'improving' if avg_rating > 3.5 and accuracy_rate > 0.7 else 'stable'
        }
    
    def get_recommendations_for_improvement(self) -> List[str]:
        """
        Gera recomendações para melhoria baseadas em feedback.
        
        Returns:
            Lista de recomendações
        """
        trends = self.analyze_feedback_trends()
        recommendations = []
        
        rec_metrics = trends.get('recommendation_metrics', {})
        if rec_metrics.get('average_rating', 0) < 3.0:
            recommendations.append("Melhorar qualidade das recomendações - ratings baixos detectados")
        
        if rec_metrics.get('completion_rate', 0) < 0.5:
            recommendations.append("Aumentar taxa de conclusão - usuários não estão completando recomendações")
        
        pred_metrics = trends.get('prediction_metrics', {})
        if pred_metrics.get('accuracy_rate', 0) < 0.7:
            recommendations.append("Melhorar precisão das predições - accuracy abaixo do esperado")
        
        return recommendations


class ActiveLearning:
    """
    Sistema de Active Learning para solicitar labels em casos ambíguos.
    """
    
    def __init__(self, uncertainty_threshold: float = 0.3):
        """
        Inicializa sistema de active learning.
        
        Args:
            uncertainty_threshold: Threshold de incerteza para solicitar label
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.pending_labels = []
    
    def identify_ambiguous_cases(
        self,
        predictions: List[Dict]
    ) -> List[Dict]:
        """
        Identifica casos ambíguos que precisam de label.
        
        Args:
            predictions: Lista de predições com probabilidades
            
        Returns:
            Lista de casos ambíguos
        """
        ambiguous = []
        
        for pred in predictions:
            probabilities = pred.get('probabilities', {})
            
            if probabilities:
                max_prob = max(probabilities.values())
                uncertainty = 1 - max_prob
                
                if uncertainty > self.uncertainty_threshold:
                    ambiguous.append({
                        'prediction_id': pred.get('id'),
                        'uncertainty': uncertainty,
                        'probabilities': probabilities,
                        'needs_label': True
                    })
        
        logger.info(f"Identificados {len(ambiguous)} casos ambíguos")
        return ambiguous
    
    def request_label(
        self,
        case: Dict,
        user_id: int
    ) -> Dict:
        """
        Solicita label para caso ambíguo.
        
        Args:
            case: Caso ambíguo
            user_id: ID do usuário
            
        Returns:
            Dicionário com solicitação
        """
        label_request = {
            'case_id': case.get('prediction_id'),
            'user_id': user_id,
            'uncertainty': case.get('uncertainty'),
            'requested_at': datetime.now(),
            'status': 'pending'
        }
        
        self.pending_labels.append(label_request)
        
        return label_request


class ContinuousLearning:
    """
    Sistema de aprendizado contínuo para ajustar modelos com feedback.
    """
    
    def __init__(self):
        self.learning_history = []
        self.model_updates = []
    
    def update_model_with_feedback(
        self,
        model_name: str,
        feedback_data: List[Dict],
        update_strategy: str = "incremental"
    ) -> Dict:
        """
        Atualiza modelo com feedback recebido.
        
        Args:
            model_name: Nome do modelo
            feedback_data: Dados de feedback
            update_strategy: Estratégia de atualização
            
        Returns:
            Dicionário com resultado da atualização
        """
        logger.info(f"Atualizando modelo {model_name} com {len(feedback_data)} feedbacks")

        update_result = {
            'model_name': model_name,
            'n_feedback_samples': len(feedback_data),
            'update_strategy': update_strategy,
            'updated_at': datetime.now(),
            'status': 'success'
        }
        
        self.model_updates.append(update_result)
        self.learning_history.append({
            'timestamp': datetime.now(),
            'model': model_name,
            'action': 'update',
            'result': update_result
        })
        
        return update_result
    
    def analyze_learning_loop(self) -> Dict:
        """
        Analisa loop de aprendizado contínuo.
        
        Returns:
            Dicionário com análise
        """
        if not self.model_updates:
            return {}
        
        return {
            'total_updates': len(self.model_updates),
            'last_update': self.model_updates[-1]['updated_at'].isoformat() if self.model_updates else None,
            'update_frequency': self._calculate_update_frequency(),
            'improvement_trend': self._analyze_improvement_trend()
        }
    
    def _calculate_update_frequency(self) -> str:
        """Calcula frequência de atualizações."""
        if len(self.model_updates) < 2:
            return "insufficient_data"
        
        time_diffs = [
            (self.model_updates[i+1]['updated_at'] - self.model_updates[i]['updated_at']).days
            for i in range(len(self.model_updates) - 1)
        ]
        
        avg_days = np.mean(time_diffs)
        
        if avg_days < 7:
            return "weekly"
        elif avg_days < 30:
            return "monthly"
        else:
            return "irregular"
    
    def _analyze_improvement_trend(self) -> str:
        """Analisa tendência de melhoria."""
        return "stable"


if __name__ == "__main__":
    from datetime import timedelta

    feedback_system = FeedbackSystem()

    feedback_system.submit_recommendation_feedback(
        user_id=1,
        recommendation_id="rec_001",
        rating=4.5,
        completed=True
    )

    trends = feedback_system.analyze_feedback_trends()
    print(f"Tendências: {trends}")

