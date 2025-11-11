"""Sistema de recomendação híbrido baseado em conteúdo e feedback implícito."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Estrutura para recomendação."""
    item_id: str
    item_type: str  # 'exercise', 'meditation', 'break_time', 'content'
    title: str
    description: str
    score: float
    reason: str


class RecommendationEngine:
    """
    Engine de recomendação híbrida combinando:
    - Collaborative Filtering
    - Content-Based Filtering
    - Reinforcement Learning (Multi-Armed Bandit)
    """
    
    def __init__(self):
        # Dados de usuários e interações
        self.user_interactions = defaultdict(list)  # {user_id: [interactions]}
        self.item_features = {}
        self.user_profiles = {}
        
        # Modelos
        self.collaborative_model = None
        self.content_model = None
        
        # Multi-Armed Bandit
        self.bandit_arms = {}  # {item_id: {count, reward_sum}}
        
        # Catálogo de itens recomendáveis
        self.items_catalog = self._initialize_catalog()
        
    def _initialize_catalog(self) -> Dict:
        """Inicializa catálogo de itens recomendáveis."""
        return {
            'mindfulness_exercises': [
                {
                    'id': 'mind_001',
                    'title': 'Respiração 4-7-8',
                    'description': 'Técnica de respiração para reduzir ansiedade',
                    'duration': 5,
                    'tags': ['ansiedade', 'stress', 'respiração'],
                    'difficulty': 'iniciante'
                },
                {
                    'id': 'mind_002',
                    'title': 'Meditação Guiada 10min',
                    'description': 'Meditação guiada para relaxamento',
                    'duration': 10,
                    'tags': ['relaxamento', 'meditação', 'mindfulness'],
                    'difficulty': 'iniciante'
                },
                {
                    'id': 'mind_003',
                    'title': 'Body Scan',
                    'description': 'Técnica de atenção plena corporal',
                    'duration': 15,
                    'tags': ['atenção plena', 'corpo', 'relaxamento'],
                    'difficulty': 'intermediário'
                }
            ],
            'break_times': [
                {
                    'id': 'break_001',
                    'title': 'Pausa Curta (5min)',
                    'description': 'Pausa rápida para recarregar',
                    'duration': 5,
                    'tags': ['curta', 'rápida'],
                    'best_time': 'manhã'
                },
                {
                    'id': 'break_002',
                    'title': 'Pausa Longa (15min)',
                    'description': 'Pausa para caminhada ou alongamento',
                    'duration': 15,
                    'tags': ['longa', 'exercício'],
                    'best_time': 'tarde'
                }
            ],
            'wellbeing_content': [
                {
                    'id': 'content_001',
                    'title': 'Artigo: Gestão de Stress',
                    'description': 'Estratégias práticas para gerenciar stress',
                    'duration': 10,
                    'tags': ['stress', 'gestão', 'artigo'],
                    'category': 'educação'
                }
            ]
        }
    
    def train_collaborative_model(self, interactions_df: pd.DataFrame):
        """
        Treina modelo de collaborative filtering.
        
        Args:
            interactions_df: DataFrame com colunas [user_id, item_id, rating, timestamp]
        """
        logger.info("Treinando modelo de collaborative filtering")
        
        # Criar matriz usuário-item
        user_item_matrix = interactions_df.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0
        )
        
        # Calcular similaridade entre usuários
        user_similarity = cosine_similarity(user_item_matrix)
        
        self.collaborative_model = {
            'user_item_matrix': user_item_matrix,
            'user_similarity': user_similarity,
            'user_ids': user_item_matrix.index.tolist()
        }
        
        logger.info("Modelo de collaborative filtering treinado")
    
    def train_content_model(self, items_df: pd.DataFrame):
        """
        Treina modelo content-based usando características dos itens.
        
        Args:
            items_df: DataFrame com características dos itens
        """
        logger.info("Treinando modelo content-based")
        
        # Combinar features textuais
        items_df['combined_features'] = (
            items_df.get('tags', '').fillna('') + ' ' +
            items_df.get('description', '').fillna('') + ' ' +
            items_df.get('title', '').fillna('')
        )
        
        # TF-IDF
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(items_df['combined_features'])
        
        # Similaridade entre itens
        item_similarity = cosine_similarity(tfidf_matrix)
        
        self.content_model = {
            'vectorizer': vectorizer,
            'item_similarity': item_similarity,
            'item_ids': items_df['id'].tolist()
        }
        
        logger.info("Modelo content-based treinado")
    
    def collaborative_recommend(
        self,
        user_id: int,
        n_recommendations: int = 5
    ) -> List[Recommendation]:
        """Gera recomendações usando collaborative filtering."""
        if self.collaborative_model is None:
            return []
        
        user_item_matrix = self.collaborative_model['user_item_matrix']
        user_similarity = self.collaborative_model['user_similarity']
        
        if user_id not in user_item_matrix.index:
            # Usuário novo - usar média geral
            return self._get_popular_items(n_recommendations)
        
        # Encontrar usuários similares
        user_idx = user_item_matrix.index.get_loc(user_id)
        similar_users_idx = np.argsort(user_similarity[user_idx])[-10:-1][::-1]
        
        # Itens que usuários similares gostaram
        recommendations = {}
        for similar_idx in similar_users_idx:
            similar_user_id = user_item_matrix.index[similar_idx]
            similarity_score = user_similarity[user_idx, similar_idx]
            
            # Itens que o usuário similar avaliou bem mas o usuário atual não
            user_items = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)
            similar_user_items = user_item_matrix.loc[similar_user_id]
            
            for item_id, rating in similar_user_items.items():
                if item_id not in user_items and rating > 3:
                    if item_id not in recommendations:
                        recommendations[item_id] = 0
                    recommendations[item_id] += rating * similarity_score
        
        # Ordenar e retornar top N
        sorted_items = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        recommendations: List[Recommendation] = []
        for item_id, score in sorted_items[:n_recommendations]:
            rec = self._create_recommendation(item_id, score, 'collaborative')
            if rec:
                recommendations.append(rec)
        return recommendations
    
    def content_based_recommend(
        self,
        user_id: int,
        user_profile: Dict,
        n_recommendations: int = 5
    ) -> List[Recommendation]:
        """Gera recomendações usando content-based filtering."""
        if self.content_model is None:
            return []
        
        # Extrair preferências do usuário
        user_tags = user_profile.get('preferred_tags', [])
        user_stress_level = user_profile.get('stress_level', 5)
        user_available_time = user_profile.get('available_time', 10)
        
        recommendations = []
        
        # Buscar itens que correspondem ao perfil
        for category, items in self.items_catalog.items():
            for item in items:
                score = 0.0
                reasons = []
                
                # Match de tags
                item_tags = item.get('tags', [])
                tag_matches = sum(1 for tag in user_tags if tag in item_tags)
                if tag_matches > 0:
                    score += tag_matches * 0.3
                    reasons.append(f"Corresponde às suas preferências")
                
                # Match de contexto (stress level)
                if user_stress_level >= 7 and 'stress' in item_tags:
                    score += 0.3
                    reasons.append("Recomendado para alto stress")
                
                # Match de duração
                item_duration = item.get('duration', 10)
                if item_duration <= user_available_time:
                    score += 0.2
                    reasons.append(f"Duração adequada ({item_duration}min)")
                
                if score > 0:
                    rec = self._create_recommendation(
                        item['id'],
                        score,
                        'content',
                        reason='; '.join(reasons)
                    )
                    if rec:
                        recommendations.append(rec)
        
        # Ordenar por score
        recommendations.sort(key=lambda x: x.score, reverse=True)
        
        return recommendations[:n_recommendations]
    
    def multi_armed_bandit_recommend(
        self,
        user_id: int,
        n_recommendations: int = 5,
        exploration_rate: float = 0.2
    ) -> List[Recommendation]:
        """
        Gera recomendações usando Multi-Armed Bandit.
        Balanceia exploração (novos itens) vs exploitation (itens conhecidos).
        """
        # Inicializar bandit arms se necessário
        all_items = []
        for items in self.items_catalog.values():
            all_items.extend(items)
        
        for item in all_items:
            if item['id'] not in self.bandit_arms:
                self.bandit_arms[item['id']] = {
                    'count': 0,
                    'reward_sum': 0.0,
                    'avg_reward': 0.5  # Prior otimista
                }
        
        # Calcular UCB (Upper Confidence Bound) para cada item
        total_pulls = sum(arm['count'] for arm in self.bandit_arms.values())
        
        item_scores = []
        for item_id, arm in self.bandit_arms.items():
            if arm['count'] == 0:
                ucb_score = 1.0
            else:
                avg_reward = arm['avg_reward']
                confidence = np.sqrt(
                    (2 * np.log(total_pulls + 1)) / arm['count']
                )
                ucb_score = avg_reward + exploration_rate * confidence
            
            item_scores.append((item_id, float(ucb_score)))
        
        # Ordenar por UCB score
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Selecionar top N com mix de exploração/exploitation
        n_explore = int(n_recommendations * exploration_rate)
        n_exploit = n_recommendations - n_explore
        
        recommendations = []
        
        # Exploitation: itens com melhor histórico
        exploit_items = sorted(
            self.bandit_arms.items(),
            key=lambda x: x[1]['avg_reward'],
            reverse=True
        )[:n_exploit]
        
        for item_id, arm in exploit_items:
            rec = self._create_recommendation(
                item_id,
                arm['avg_reward'],
                'bandit_exploit',
                reason=f"Baseado em feedback positivo (score: {arm['avg_reward']:.2f})"
            )
            if rec:
                recommendations.append(rec)
        
        # Exploração: itens menos testados mas promissores
        explore_items = [
            (item_id, score) for item_id, score in item_scores
            if item_id not in [r.item_id for r in recommendations]
        ][:n_explore]
        
        for item_id, score in explore_items:
            rec = self._create_recommendation(
                item_id,
                score,
                'bandit_explore',
                reason="Novo item para explorar"
            )
            if rec:
                recommendations.append(rec)

        return recommendations
    
    def hybrid_recommend(
        self,
        user_id: int,
        user_profile: Dict,
        context: Optional[Dict] = None,
        n_recommendations: int = 5
    ) -> List[Recommendation]:
        """
        Gera recomendações híbridas combinando todos os métodos.
        
        Args:
            user_id: ID do usuário
            user_profile: Perfil do usuário
            context: Contexto atual (stress, hora do dia, etc.)
            n_recommendations: Número de recomendações
            
        Returns:
            Lista de recomendações
        """
        # Pesos para cada método
        w_collaborative = 0.3
        w_content = 0.4
        w_bandit = 0.3
        
        # Obter recomendações de cada método
        collab_recs = self.collaborative_recommend(user_id, n_recommendations * 2)
        content_recs = self.content_based_recommend(user_id, user_profile, n_recommendations * 2)
        bandit_recs = self.multi_armed_bandit_recommend(user_id, n_recommendations * 2)
        
        # Combinar e re-ranquear
        combined_scores = defaultdict(float)
        
        for rec in collab_recs:
            combined_scores[rec.item_id] += rec.score * w_collaborative
        
        for rec in content_recs:
            combined_scores[rec.item_id] += rec.score * w_content
        
        for rec in bandit_recs:
            combined_scores[rec.item_id] += rec.score * w_bandit
        
        # Aplicar contexto temporal
        if context:
            current_hour = datetime.now().hour
            for item_id in combined_scores:
                item = self._get_item_by_id(item_id)
                if item:
                    best_time = item.get('best_time', '')
                    if best_time == 'manhã' and 6 <= current_hour < 12:
                        combined_scores[item_id] *= 1.2
                    elif best_time == 'tarde' and 12 <= current_hour < 18:
                        combined_scores[item_id] *= 1.2
        
        # Ordenar e retornar top N
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_recommendations = []
        for item_id, score in sorted_items[:n_recommendations]:
            item = self._get_item_by_id(item_id)
            if item:
                final_recommendations.append(
                    Recommendation(
                        item_id=item_id,
                        item_type=self._get_item_type(item_id),
                        title=item['title'],
                        description=item['description'],
                        score=score,
                        reason=f"Recomendação híbrida (score: {score:.2f})"
                    )
                )
        
        return final_recommendations
    
    def update_feedback(
        self,
        user_id: int,
        item_id: str,
        rating: float,
        completed: bool = True
    ):
        """
        Atualiza feedback do usuário para melhorar recomendações futuras.
        Usa reinforcement learning para ajustar estratégias.
        """
        # Atualizar interações do usuário
        self.user_interactions[user_id].append({
            'item_id': item_id,
            'rating': rating,
            'completed': completed,
            'timestamp': datetime.now()
        })
        
        # Atualizar Multi-Armed Bandit
        if item_id in self.bandit_arms:
            arm = self.bandit_arms[item_id]
            arm['count'] += 1
            reward = rating / 5.0 if completed else rating / 10.0  # Penalizar não completar
            arm['reward_sum'] += reward
            arm['avg_reward'] = arm['reward_sum'] / arm['count']
        
        logger.info(f"Feedback atualizado: usuário {user_id}, item {item_id}, rating {rating}")
    
    def _create_recommendation(
        self,
        item_id: str,
        score: float,
        method: str,
        reason: Optional[str] = None
    ) -> Optional[Recommendation]:
        """Cria objeto Recommendation para um item existente."""
        item = self._get_item_by_id(item_id)
        if not item:
            return None
        
        return Recommendation(
            item_id=item_id,
            item_type=self._get_item_type(item_id),
            title=item['title'],
            description=item['description'],
            score=score,
            reason=reason or f"Recomendado por {method}"
        )
    
    def _get_item_by_id(self, item_id: str) -> Optional[Dict]:
        """Busca item no catálogo por ID."""
        for items in self.items_catalog.values():
            for item in items:
                if item['id'] == item_id:
                    return item
        return None
    
    def _get_item_type(self, item_id: str) -> str:
        """Determina tipo do item pelo ID."""
        if item_id.startswith('mind_'):
            return 'exercise'
        elif item_id.startswith('break_'):
            return 'break_time'
        elif item_id.startswith('content_'):
            return 'content'
        return 'unknown'
    
    def _get_popular_items(self, n: int) -> List[Recommendation]:
        """Retorna itens populares ou padrão quando não há histórico."""
        if not self.bandit_arms:
            defaults: List[Recommendation] = []
            for items in self.items_catalog.values():
                for item in items:
                    rec = self._create_recommendation(
                        item['id'],
                        score=0.5,
                        method='default',
                        reason='Sugestão inicial'
                    )
                    if rec:
                        defaults.append(rec)
            return defaults[:n]

        popular = sorted(
            self.bandit_arms.items(),
            key=lambda x: x[1]['avg_reward'],
            reverse=True
        )[:n]

        recommendations: List[Recommendation] = []
        for item_id, arm in popular:
            rec = self._create_recommendation(item_id, arm['avg_reward'], 'popular')
            if rec:
                recommendations.append(rec)
        return recommendations


if __name__ == "__main__":
    # Exemplo de uso
    engine = RecommendationEngine()
    
    user_profile = {
        'preferred_tags': ['stress', 'relaxamento'],
        'stress_level': 8,
        'available_time': 10
    }
    
    recommendations = engine.hybrid_recommend(
        user_id=1,
        user_profile=user_profile,
        n_recommendations=5
    )
    
    for rec in recommendations:
        print(f"{rec.title}: {rec.score:.2f} - {rec.reason}")

