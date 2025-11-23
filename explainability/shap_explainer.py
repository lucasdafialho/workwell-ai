"""
Explicabilidade de modelos usando SHAP (SHapley Additive exPlanations) e LIME.
"""

import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
import matplotlib.pyplot as plt
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Classe para explicar predições de modelos usando SHAP e LIME.
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Inicializa explainer.
        
        Args:
            model: Modelo treinado
            feature_names: Nomes das features
        """
        self.model = model
        self.feature_names = feature_names
    
    def explain_with_shap(
        self,
        X: np.ndarray,
        background_data: Optional[np.ndarray] = None,
        max_evals: int = 100
    ) -> Dict:
        """
        Explica predições usando SHAP.
        
        Args:
            X: Dados para explicar
            background_data: Dados de background para SHAP
            max_evals: Número máximo de avaliações
            
        Returns:
            Dicionário com explicações
        """
        logger.info("Gerando explicações SHAP")

        if background_data is None:
            background_data = X[:100]  

        if isinstance(self.model, torch.nn.Module):
            def model_wrapper(x):
                self.model.eval()
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x)
                    outputs, _ = self.model(x_tensor)
                    return outputs.numpy()
            
            explainer = shap.KernelExplainer(model_wrapper, background_data)
        else:
            # Para modelos scikit-learn
            explainer = shap.KernelExplainer(self.model.predict_proba, background_data)
        
        # Calcular SHAP values
        shap_values = explainer.shap_values(X[:10], nsamples=max_evals)  # Limitar para performance
        
        # Processar resultados
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Classe positiva
        
        # Calcular importância média das features
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        explanations = {
            'shap_values': shap_values.tolist(),
            'feature_importance': dict(zip(self.feature_names, feature_importance.tolist())),
            'top_features': [
                {'feature': name, 'importance': float(imp)}
                for name, imp in sorted(
                    zip(self.feature_names, feature_importance),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            ]
        }
        
        logger.info("Explicações SHAP geradas")
        return explanations
    
    def explain_with_lime(
        self,
        X: np.ndarray,
        instance_idx: int = 0,
        num_features: int = 10
    ) -> Dict:
        """
        Explica predição específica usando LIME.
        
        Args:
            X: Dados
            instance_idx: Índice da instância para explicar
            num_features: Número de features a mostrar
            
        Returns:
            Dicionário com explicação
        """
        logger.info(f"Gerando explicação LIME para instância {instance_idx}")
        
        # Criar explainer LIME
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X,
            feature_names=self.feature_names,
            mode='regression' if not hasattr(self.model, 'predict_proba') else 'classification'
        )
        
        # Função wrapper para o modelo
        def model_predict(x):
            if isinstance(self.model, torch.nn.Module):
                self.model.eval()
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x)
                    outputs, _ = self.model(x_tensor)
                    return outputs.numpy()
            else:
                return self.model.predict_proba(x)
        
        # Gerar explicação
        explanation = explainer.explain_instance(
            X[instance_idx],
            model_predict,
            num_features=num_features
        )
        
        # Processar resultados
        exp_list = explanation.as_list()
        
        explanations = {
            'instance_idx': instance_idx,
            'prediction': float(explanation.predicted_value),
            'local_explanation': [
                {'feature': feat, 'weight': float(weight)}
                for feat, weight in exp_list
            ],
            'top_positive_features': [
                {'feature': feat, 'weight': float(weight)}
                for feat, weight in exp_list if weight > 0
            ][:5],
            'top_negative_features': [
                {'feature': feat, 'weight': float(weight)}
                for feat, weight in exp_list if weight < 0
            ][:5]
        }
        
        logger.info("Explicação LIME gerada")
        return explanations
    
    def generate_explanation_text(
        self,
        prediction: Dict,
        shap_explanation: Dict,
        lime_explanation: Optional[Dict] = None
    ) -> str:
        """
        Gera texto em linguagem natural explicando a predição.
        
        Args:
            prediction: Resultado da predição
            shap_explanation: Explicação SHAP
            lime_explanation: Explicação LIME (opcional)
            
        Returns:
            Texto explicativo
        """
        predicted_class = prediction.get('predicted_class', 'desconhecido')
        probabilities = prediction.get('probabilities', {})
        
        text = f"O modelo previu um risco de burnout '{predicted_class}' "
        text += f"com {probabilities.get(predicted_class, 0)*100:.1f}% de confiança.\n\n"
        
        # Features mais importantes
        top_features = shap_explanation.get('top_features', [])[:5]
        if top_features:
            text += "Os fatores mais importantes para esta predição foram:\n"
            for i, feat in enumerate(top_features, 1):
                text += f"{i}. {feat['feature']}: contribuição de {feat['importance']:.3f}\n"
        
        # Explicação LIME se disponível
        if lime_explanation:
            text += "\nExplicação local:\n"
            positive = lime_explanation.get('top_positive_features', [])
            negative = lime_explanation.get('top_negative_features', [])
            
            if positive:
                text += "Fatores que aumentam o risco:\n"
                for feat in positive:
                    text += f"- {feat['feature']} (peso: {feat['weight']:.3f})\n"
            
            if negative:
                text += "\nFatores que reduzem o risco:\n"
                for feat in negative:
                    text += f"- {feat['feature']} (peso: {feat['weight']:.3f})\n"
        
        return text
    
    def create_visualization(
        self,
        shap_explanation: Dict,
        output_path: str
    ):
        """
        Cria visualização das explicações.
        
        Args:
            shap_explanation: Explicação SHAP
            output_path: Caminho para salvar imagem
        """
        feature_importance = shap_explanation.get('feature_importance', {})
        
        if not feature_importance:
            return
        
        # Ordenar por importância
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        features, importances = zip(*sorted_features)
        
        # Criar gráfico
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importância (SHAP)')
        plt.title('Importância das Features para Predição de Burnout')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualização salva em {output_path}")
    
    def counterfactual_explanation(
        self,
        instance: np.ndarray,
        target_class: str,
        feature_ranges: Dict[str, tuple]
    ) -> Dict:
        """
        Gera explicação contrafactual mostrando o que mudaria o resultado.
        
        Args:
            instance: Instância atual
            target_class: Classe desejada
            feature_ranges: Intervalos possíveis para cada feature
            
        Returns:
            Dicionário com explicação contrafactual
        """
        logger.info("Gerando explicação contrafactual")
        
        # Simulação simples - em produção, usar otimização
        current_pred = self._predict_single(instance)
        
        changes = []
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in feature_ranges:
                min_val, max_val = feature_ranges[feature_name]
                current_val = instance[i]
                
                # Testar mudança para valor médio
                test_instance = instance.copy()
                test_instance[i] = (min_val + max_val) / 2
                test_pred = self._predict_single(test_instance)
                
                if test_pred != current_pred:
                    changes.append({
                        'feature': feature_name,
                        'current_value': float(current_val),
                        'suggested_value': float((min_val + max_val) / 2),
                        'impact': 'mudaria predição'
                    })
        
        return {
            'current_prediction': current_pred,
            'target_class': target_class,
            'suggested_changes': changes[:5]  # Top 5 mudanças
        }
    
    def _predict_single(self, instance: np.ndarray) -> str:
        """Faz predição para instância única."""
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(instance).unsqueeze(0)
                outputs, probabilities = self.model(x_tensor)
                _, predicted = torch.max(outputs, 1)
                return f"class_{predicted.item()}"
        else:
            pred = self.model.predict(instance.reshape(1, -1))[0]
            return str(pred)


if __name__ == "__main__":
    # Exemplo de uso
    # (requer modelo treinado)
    pass

