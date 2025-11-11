"""
Privacidade e Segurança de Dados.
Implementa differential privacy, federated learning e anonimização.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from diffprivlib.mechanisms import Laplace, GaussianMechanism
from diffprivlib.models import GaussianNB
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrivacyProtection:
    """
    Classe para proteção de privacidade usando differential privacy.
    """
    
    def __init__(self, epsilon: float = 1.0):
        """
        Inicializa proteção de privacidade.
        
        Args:
            epsilon: Parâmetro de privacidade (menor = mais privacidade)
        """
        self.epsilon = epsilon
        self.laplace_mechanism = Laplace(epsilon=epsilon)
        self.gaussian_mechanism = GaussianMechanism(epsilon=epsilon, delta=1e-5)
    
    def add_noise_to_aggregates(
        self,
        aggregate_value: float,
        sensitivity: float = 1.0
    ) -> float:
        """
        Adiciona ruído Laplace a agregações para differential privacy.
        
        Args:
            aggregate_value: Valor agregado
            sensitivity: Sensibilidade da função
            
        Returns:
            Valor com ruído adicionado
        """
        noise = self.laplace_mechanism.randomise(sensitivity)
        return aggregate_value + noise
    
    def anonymize_data(
        self,
        data: Dict,
        fields_to_anonymize: List[str]
    ) -> Dict:
        """
        Anonimiza dados antes do treinamento.
        
        Args:
            data: Dicionário com dados
            fields_to_anonymize: Campos para anonimizar
            
        Returns:
            Dados anonimizados
        """
        anonymized = data.copy()
        
        for field in fields_to_anonymize:
            if field in anonymized:
                # Hash do valor
                if isinstance(anonymized[field], str):
                    anonymized[field] = hashlib.sha256(
                        anonymized[field].encode()
                    ).hexdigest()[:16]
                elif isinstance(anonymized[field], (int, float)):
                    # Adicionar ruído
                    anonymized[field] = self.add_noise_to_aggregates(
                        float(anonymized[field])
                    )
        
        return anonymized
    
    def k_anonymity_check(
        self,
        dataset: List[Dict],
        quasi_identifiers: List[str],
        k: int = 3
    ) -> bool:
        """
        Verifica se dataset satisfaz k-anonimidade.
        
        Args:
            dataset: Lista de registros
            quasi_identifiers: Identificadores quase-identificadores
            k: Valor de k-anonimidade
            
        Returns:
            True se satisfaz k-anonimidade
        """
        # Agrupar por quasi-identifiers
        groups = {}
        for record in dataset:
            key = tuple(record.get(qi, None) for qi in quasi_identifiers)
            if key not in groups:
                groups[key] = []
            groups[key].append(record)
        
        # Verificar se todos os grupos têm pelo menos k elementos
        min_group_size = min(len(group) for group in groups.values())
        
        return min_group_size >= k


class FederatedLearning:
    """
    Implementação básica de Federated Learning para treinar modelos
    sem centralizar dados sensíveis.
    """
    
    def __init__(self, n_clients: int = 5):
        """
        Inicializa sistema de federated learning.
        
        Args:
            n_clients: Número de clientes federados
        """
        self.n_clients = n_clients
        self.client_models = {}
        self.global_model = None
    
    def train_federated_round(
        self,
        client_data: Dict[int, np.ndarray],
        model_class,
        n_epochs: int = 5
    ) -> Dict:
        """
        Executa uma rodada de treinamento federado.
        
        Args:
            client_data: Dicionário {client_id: data}
            model_class: Classe do modelo
            n_epochs: Número de épocas por cliente
            
        Returns:
            Dicionário com métricas da rodada
        """
        logger.info(f"Executando rodada federada com {len(client_data)} clientes")
        
        client_weights = {}
        
        # Treinar modelo em cada cliente
        for client_id, data in client_data.items():
            logger.info(f"Treinando cliente {client_id}")
            
            # Criar modelo local
            local_model = model_class()
            
            # Treinar localmente (simulação)
            # Em produção, isso rodaria no dispositivo do cliente
            local_model.fit(data)
            
            # Obter pesos do modelo
            if hasattr(local_model, 'coef_'):
                client_weights[client_id] = {
                    'weights': local_model.coef_,
                    'n_samples': len(data)
                }
        
        # Agregar pesos (FedAvg)
        aggregated_weights = self._federated_averaging(client_weights)
        
        # Atualizar modelo global
        if self.global_model is None:
            self.global_model = model_class()
        
        # Aplicar pesos agregados
        if hasattr(self.global_model, 'coef_'):
            self.global_model.coef_ = aggregated_weights
        
        return {
            'n_clients': len(client_data),
            'aggregation_method': 'FedAvg',
            'status': 'completed'
        }
    
    def _federated_averaging(self, client_weights: Dict) -> np.ndarray:
        """
        Agrega pesos usando FedAvg (Federated Averaging).
        
        Args:
            client_weights: Dicionário {client_id: {weights, n_samples}}
            
        Returns:
            Pesos agregados
        """
        total_samples = sum(w['n_samples'] for w in client_weights.values())
        
        # Média ponderada por número de amostras
        aggregated = None
        
        for client_id, weights_data in client_weights.items():
            weight = weights_data['weights']
            n_samples = weights_data['n_samples']
            
            if aggregated is None:
                aggregated = weight * (n_samples / total_samples)
            else:
                aggregated += weight * (n_samples / total_samples)
        
        return aggregated


class DataEncryption:
    """
    Utilitários para criptografia de dados e modelos.
    """
    
    @staticmethod
    def encrypt_model_weights(weights: np.ndarray, key: str) -> bytes:
        """
        Criptografa pesos do modelo.
        
        Args:
            weights: Pesos do modelo
            key: Chave de criptografia
            
        Returns:
            Pesos criptografados
        """
        # Simulação - em produção, usar biblioteca de criptografia adequada
        import pickle
        serialized = pickle.dumps(weights)
        
        # Hash simples (em produção, usar AES)
        key_hash = hashlib.sha256(key.encode()).digest()
        encrypted = bytes(a ^ b for a, b in zip(serialized, key_hash * (len(serialized) // len(key_hash) + 1)))
        
        return encrypted
    
    @staticmethod
    def decrypt_model_weights(encrypted: bytes, key: str) -> np.ndarray:
        """
        Descriptografa pesos do modelo.
        
        Args:
            encrypted: Pesos criptografados
            key: Chave de descriptografia
            
        Returns:
            Pesos descriptografados
        """
        import pickle
        
        # Descriptografar
        key_hash = hashlib.sha256(key.encode()).digest()
        decrypted = bytes(a ^ b for a, b in zip(encrypted, key_hash * (len(encrypted) // len(key_hash) + 1)))
        
        return pickle.loads(decrypted)


class AccessControl:
    """
    Sistema de controle de acesso granular para dados sensíveis.
    """
    
    def __init__(self):
        self.access_logs = []
        self.permissions = {}
    
    def check_access(
        self,
        user_id: int,
        resource_type: str,
        action: str
    ) -> bool:
        """
        Verifica se usuário tem permissão para ação.
        
        Args:
            user_id: ID do usuário
            resource_type: Tipo de recurso
            action: Ação solicitada
            
        Returns:
            True se tem permissão
        """
        permission_key = f"{user_id}:{resource_type}:{action}"
        
        # Log de acesso
        self.access_logs.append({
            'user_id': user_id,
            'resource_type': resource_type,
            'action': action,
            'timestamp': datetime.now(),
            'granted': permission_key in self.permissions
        })
        
        return permission_key in self.permissions
    
    def grant_permission(
        self,
        user_id: int,
        resource_type: str,
        action: str
    ):
        """Concede permissão."""
        permission_key = f"{user_id}:{resource_type}:{action}"
        self.permissions[permission_key] = True
    
    def get_access_logs(self, user_id: Optional[int] = None) -> List[Dict]:
        """Retorna logs de acesso."""
        if user_id:
            return [log for log in self.access_logs if log['user_id'] == user_id]
        return self.access_logs


if __name__ == "__main__":
    # Exemplo de uso
    privacy = PrivacyProtection(epsilon=1.0)
    
    # Adicionar ruído a agregação
    original_value = 50.5
    noisy_value = privacy.add_noise_to_aggregates(original_value)
    print(f"Original: {original_value}, Com ruído: {noisy_value}")

