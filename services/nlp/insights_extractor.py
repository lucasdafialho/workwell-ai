"""
NLP Avançado para extração de insights, topic modeling e análise de texto.
Usa Named Entity Recognition, LDA e BERT-topic.
"""

import spacy
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter
import logging
from datetime import datetime
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLPInsightsExtractor:
    """
    Extrator de insights usando técnicas avançadas de NLP.
    """
    
    def __init__(self, language: str = "pt"):
        """
        Inicializa extrator de insights.
        
        Args:
            language: Idioma ('pt' para português)
        """
        self.language = language
        
        # Carregar modelo spaCy
        try:
            if language == "pt":
                self.nlp = spacy.load("pt_core_news_sm")
            else:
                self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Modelo spaCy não encontrado. Execute: python -m spacy download pt_core_news_sm")
            self.nlp = None
        
        # Padrões para detecção
        self.stress_patterns = [
            r'\b(sobrecarregado|sobrecarga|estressado|pressão|deadline|urgente)\b',
            r'\b(não consigo|não dá|impossível|difícil demais)\b',
            r'\b(cansado demais|exausto|esgotado|sem energia)\b'
        ]
        
        self.satisfaction_patterns = [
            r'\b(satisfeito|realizado|orgulhoso|feliz|contente)\b',
            r'\b(bom trabalho|ótimo|excelente|bem)\b'
        ]
    
    def extract_named_entities(self, texts: List[str]) -> Dict:
        """
        Extrai entidades nomeadas (projetos, pessoas, deadlines).
        
        Args:
            texts: Lista de textos
            
        Returns:
            Dicionário com entidades extraídas
        """
        if self.nlp is None:
            return {}
        
        logger.info("Extraindo entidades nomeadas")
        
        all_entities = {
            'PERSON': [],
            'ORG': [],
            'PROJECT': [],
            'DATE': [],
            'MONEY': []
        }
        
        for text in texts:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    all_entities['PERSON'].append(ent.text)
                elif ent.label_ == 'ORG':
                    all_entities['ORG'].append(ent.text)
                elif ent.label_ == 'DATE':
                    all_entities['DATE'].append(ent.text)
                elif ent.label_ == 'MONEY':
                    all_entities['MONEY'].append(ent.text)
                
                # Detectar projetos (padrão customizado)
                if 'projeto' in ent.text.lower() or 'project' in ent.text.lower():
                    all_entities['PROJECT'].append(ent.text)
        
        # Contar frequências
        entity_counts = {
            'people': Counter(all_entities['PERSON']).most_common(10),
            'organizations': Counter(all_entities['ORG']).most_common(10),
            'projects': Counter(all_entities['PROJECT']).most_common(10),
            'dates': Counter(all_entities['DATE']).most_common(10)
        }
        
        return {
            'entities': all_entities,
            'counts': entity_counts,
            'total_entities': sum(len(v) for v in all_entities.values())
        }
    
    def topic_modeling_lda(self, texts: List[str], n_topics: int = 5) -> Dict:
        """
        Descobre temas recorrentes usando LDA.
        
        Args:
            texts: Lista de textos
            n_topics: Número de tópicos a descobrir
            
        Returns:
            Dicionário com tópicos e palavras-chave
        """
        logger.info(f"Executando topic modeling com LDA ({n_topics} tópicos)")
        
        # Preparar textos
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Vectorização
        vectorizer = CountVectorizer(
            max_features=100,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        doc_term_matrix = vectorizer.fit_transform(processed_texts)
        
        # LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        lda.fit(doc_term_matrix)
        
        # Extrair tópicos
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_weights = [topic[i] for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weights': top_weights.tolist(),
                'name': self._name_topic(top_words)
            })
        
        # Distribuição de tópicos por documento
        doc_topics = lda.transform(doc_term_matrix)
        
        return {
            'topics': topics,
            'document_topic_distribution': doc_topics.tolist(),
            'n_topics': n_topics
        }
    
    def detect_overload_language(self, texts: List[str]) -> Dict:
        """
        Detecta linguagem indicativa de sobrecarga ou insatisfação.
        
        Args:
            texts: Lista de textos
            
        Returns:
            Dicionário com detecções
        """
        logger.info("Detectando linguagem de sobrecarga")
        
        overload_indicators = []
        satisfaction_indicators = []
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            
            # Verificar padrões de sobrecarga
            for pattern in self.stress_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    overload_indicators.append({
                        'text_index': i,
                        'text_snippet': text[:100],
                        'pattern_matched': pattern,
                        'severity': 'high' if 'demais' in text_lower or 'impossível' in text_lower else 'medium'
                    })
                    break
            
            # Verificar padrões de satisfação
            for pattern in self.satisfaction_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    satisfaction_indicators.append({
                        'text_index': i,
                        'text_snippet': text[:100],
                        'pattern_matched': pattern
                    })
                    break
        
        return {
            'overload_detected': len(overload_indicators) > 0,
            'overload_count': len(overload_indicators),
            'overload_indicators': overload_indicators[:10],
            'satisfaction_detected': len(satisfaction_indicators) > 0,
            'satisfaction_count': len(satisfaction_indicators),
            'satisfaction_indicators': satisfaction_indicators[:10],
            'overload_ratio': len(overload_indicators) / len(texts) if texts else 0
        }
    
    def extract_implicit_support_needs(self, texts: List[str]) -> List[Dict]:
        """
        Extrai requisitos implícitos de suporte.
        
        Args:
            texts: Lista de textos
            
        Returns:
            Lista de necessidades de suporte detectadas
        """
        logger.info("Extraindo necessidades implícitas de suporte")
        
        support_keywords = {
            'time_management': ['tempo', 'prazo', 'deadline', 'urgente', 'atrasado'],
            'workload': ['sobrecarga', 'muito trabalho', 'não dá conta', 'acumulado'],
            'emotional_support': ['cansado', 'desanimado', 'frustrado', 'ansioso'],
            'skills': ['não sei', 'dificuldade', 'preciso aprender', 'treinamento'],
            'resources': ['falta', 'preciso de', 'sem', 'não tenho']
        }
        
        support_needs = []
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            detected_needs = []
            
            for need_type, keywords in support_keywords.items():
                matches = [kw for kw in keywords if kw in text_lower]
                if matches:
                    detected_needs.append({
                        'type': need_type,
                        'keywords_found': matches,
                        'confidence': len(matches) / len(keywords)
                    })
            
            if detected_needs:
                support_needs.append({
                    'text_index': i,
                    'text_snippet': text[:200],
                    'needs': detected_needs,
                    'priority': 'high' if any(n['confidence'] > 0.5 for n in detected_needs) else 'medium'
                })
        
        return support_needs
    
    def generate_team_pattern_summary(self, texts_by_user: Dict[int, List[str]]) -> Dict:
        """
        Gera resumo automático de padrões de equipe.
        
        Args:
            texts_by_user: Dicionário {user_id: [texts]}
            
        Returns:
            Dicionário com resumo de padrões
        """
        logger.info("Gerando resumo de padrões de equipe")
        
        team_insights = {
            'total_users': len(texts_by_user),
            'total_texts': sum(len(texts) for texts in texts_by_user.values()),
            'common_themes': [],
            'user_sentiment_distribution': {},
            'overload_by_user': {}
        }
        
        # Analisar cada usuário
        all_texts = []
        for user_id, texts in texts_by_user.items():
            all_texts.extend(texts)
            
            # Detectar sobrecarga por usuário
            overload = self.detect_overload_language(texts)
            team_insights['overload_by_user'][user_id] = {
                'overload_ratio': overload['overload_ratio'],
                'overload_count': overload['overload_count']
            }
        
        # Tópicos comuns
        if len(all_texts) >= 5:
            topics = self.topic_modeling_lda(all_texts, n_topics=5)
            team_insights['common_themes'] = [
                {
                    'name': topic['name'],
                    'keywords': topic['words'][:5]
                }
                for topic in topics['topics']
            ]
        
        return team_insights
    
    def correlate_language_burnout(
        self,
        texts: List[str],
        burnout_scores: List[float]
    ) -> Dict:
        """
        Identifica correlações entre linguagem usada e níveis de burnout.
        
        Args:
            texts: Lista de textos
            burnout_scores: Scores de burnout correspondentes
            
        Returns:
            Dicionário com correlações identificadas
        """
        logger.info("Analisando correlações linguagem-burnout")
        
        # Agrupar textos por nível de burnout
        high_burnout_texts = [
            texts[i] for i, score in enumerate(burnout_scores) if score >= 75
        ]
        low_burnout_texts = [
            texts[i] for i, score in enumerate(burnout_scores) if score < 50
        ]
        
        # Extrair palavras-chave de cada grupo
        high_burnout_words = self._extract_keywords(high_burnout_texts)
        low_burnout_words = self._extract_keywords(low_burnout_texts)
        
        # Palavras distintivas
        high_only = set(high_burnout_words.keys()) - set(low_burnout_words.keys())
        low_only = set(low_burnout_words.keys()) - set(high_burnout_words.keys())
        
        return {
            'high_burnout_keywords': dict(list(high_burnout_words.items())[:20]),
            'low_burnout_keywords': dict(list(low_burnout_words.items())[:20]),
            'distinctive_high_burnout': list(high_only)[:10],
            'distinctive_low_burnout': list(low_only)[:10],
            'correlation_insights': self._generate_correlation_insights(
                high_burnout_words, low_burnout_words
            )
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocessa texto para análise."""
        # Remover caracteres especiais
        text = re.sub(r'[^\w\s]', '', text)
        # Converter para minúsculas
        text = text.lower()
        return text
    
    def _name_topic(self, words: List[str]) -> str:
        """Gera nome descritivo para tópico baseado em palavras-chave."""
        # Mapear palavras comuns para nomes de tópicos
        topic_mapping = {
            'trabalho': 'Trabalho e Projetos',
            'stress': 'Stress e Sobrecarga',
            'equipe': 'Colaboração e Equipe',
            'tempo': 'Gestão de Tempo',
            'saúde': 'Saúde e Bem-estar'
        }
        
        for word in words:
            for key, name in topic_mapping.items():
                if key in word:
                    return name
        
        return f"Tópico: {', '.join(words[:3])}"
    
    def _extract_keywords(self, texts: List[str], top_n: int = 20) -> Dict[str, int]:
        """Extrai palavras-chave mais frequentes."""
        if self.nlp is None:
            # Fallback simples
            all_words = []
            for text in texts:
                words = re.findall(r'\b\w+\b', text.lower())
                all_words.extend(words)
            return dict(Counter(all_words).most_common(top_n))
        
        all_words = []
        for text in texts:
            doc = self.nlp(text)
            words = [
                token.text.lower()
                for token in doc
                if not token.is_stop and not token.is_punct and token.is_alpha
            ]
            all_words.extend(words)
        
        return dict(Counter(all_words).most_common(top_n))
    
    def _generate_correlation_insights(
        self,
        high_words: Dict[str, int],
        low_words: Dict[str, int]
    ) -> List[str]:
        """Gera insights sobre correlações."""
        insights = []
        
        # Palavras mais frequentes em alto burnout
        top_high = sorted(high_words.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_high:
            insights.append(
                f"Palavras mais associadas a alto burnout: {', '.join([w[0] for w in top_high])}"
            )
        
        # Palavras mais frequentes em baixo burnout
        top_low = sorted(low_words.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_low:
            insights.append(
                f"Palavras mais associadas a baixo burnout: {', '.join([w[0] for w in top_low])}"
            )
        
        return insights


if __name__ == "__main__":
    # Exemplo de uso
    extractor = NLPInsightsExtractor()
    
    sample_texts = [
        "Estou muito sobrecarregado com o projeto X. O deadline é impossível.",
        "Trabalhei 12 horas hoje e ainda não terminei tudo.",
        "Estou satisfeito com o progresso da equipe esta semana."
    ]
    
    # Extrair entidades
    entities = extractor.extract_named_entities(sample_texts)
    print(f"Entidades: {entities}")
    
    # Detectar sobrecarga
    overload = extractor.detect_overload_language(sample_texts)
    print(f"Sobrecarga detectada: {overload['overload_detected']}")

