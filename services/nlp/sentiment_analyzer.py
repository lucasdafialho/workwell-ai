"""
Análise de Sentimento Avançada usando BERT ou RoBERTa fine-tunado em português.
Implementa classificação multi-label para detectar múltiplas emoções simultaneamente.
"""

import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Tornar wordcloud e matplotlib opcionais
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("wordcloud não disponível. Funcionalidade de word cloud desabilitada.")

logging.basicConfig(level=logging.INFO)
if 'logger' not in locals():
    logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analisador de sentimento avançado usando modelos transformer.
    Suporta análise multi-label e detecção de aspectos específicos.
    """
    
    def __init__(
        self,
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        use_gpu: bool = True
    ):
        """
        Inicializa analisador de sentimento.
        
        Args:
            model_name: Nome do modelo HuggingFace
            use_gpu: Se deve usar GPU se disponível
        """
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self._model_loaded = False
        self.sentiment_pipeline = None

        self.positive_keywords = ["bom", "excelente", "feliz", "animado", "produtivo", "satisfeito", "tranquilo"]
        self.negative_keywords = ["ruim", "péssimo", "cansado", "estressado", "ansioso", "desmotivado", "sobrecarregado"]

        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info("Carregando modelo transformer de sentimento (%s)...", model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=3  # positivo, neutro, negativo
                ).to(self.device)
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device.type == 'cuda' else -1
                )
                self._model_loaded = True
                logger.info("Modelo transformer carregado com sucesso.")
            except Exception as exc:
                logger.warning("Falha ao carregar modelo transformer (%s). Usando heurísticas simples. Erro: %s", model_name, exc)
                self.sentiment_pipeline = None
        else:
            logger.warning("Transformers não disponível. Usando heurísticas simples para sentimento.")
            self.sentiment_pipeline = None
        
        # Emoções específicas para análise multi-label
        self.emotions = [
            'alegria', 'tristeza', 'raiva', 'medo', 'ansiedade',
            'esperança', 'frustração', 'satisfação', 'cansaço', 'motivação'
        ]
        
        # Aspectos para análise
        self.aspects = {
            'trabalho': ['trabalho', 'projeto', 'deadline', 'reunião', 'cliente'],
            'equipe': ['colega', 'equipe', 'chefe', 'gerente', 'colaboração'],
            'vida_pessoal': ['família', 'pessoal', 'tempo', 'equilíbrio', 'descanso'],
            'saúde': ['sono', 'cansaço', 'energia', 'saúde', 'bem-estar']
        }
        
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analisa sentimento básico do texto.
        
        Args:
            text: Texto para análise
            
        Returns:
            Dicionário com sentimento e score
        """
        if not text or len(text.strip()) == 0:
            return {
                'sentiment': 'neutro',
                'score': 0.5,
                'confidence': 0.0
            }
        
        if self.sentiment_pipeline:
            result = self.sentiment_pipeline(text)[0]

            label_mapping = {
                'POSITIVE': 'positivo',
                'NEGATIVE': 'negativo',
                'NEUTRAL': 'neutro',
                'LABEL_0': 'negativo',
                'LABEL_1': 'neutro',
                'LABEL_2': 'positivo'
            }

            label = result.get('label', 'NEUTRAL')
            sentiment = label_mapping.get(label, 'neutro')
            score = result.get('score', 0.5)

            return {
                'sentiment': sentiment,
                'score': score,
                'confidence': score
            }

        return self._heuristic_sentiment(text)
    
    def analyze_multi_emotion(self, text: str) -> Dict:
        """
        Detecta múltiplas emoções simultaneamente no texto.
        
        Args:
            text: Texto para análise
            
        Returns:
            Dicionário com emoções detectadas e scores
        """
        # Palavras-chave para cada emoção
        emotion_keywords = {
            'alegria': ['feliz', 'alegre', 'satisfeito', 'contente', 'animado'],
            'tristeza': ['triste', 'deprimido', 'melancólico', 'desanimado'],
            'raiva': ['irritado', 'bravo', 'frustrado', 'nervoso', 'raiva'],
            'medo': ['medo', 'preocupado', 'ansioso', 'tenso'],
            'ansiedade': ['ansioso', 'preocupado', 'nervoso', 'agitado'],
            'esperança': ['esperança', 'otimista', 'confiante', 'positivo'],
            'frustração': ['frustrado', 'desapontado', 'insatisfeito'],
            'satisfação': ['satisfeito', 'realizado', 'orgulhoso', 'conteúdo'],
            'cansaço': ['cansado', 'exausto', 'fatigado', 'esgotado'],
            'motivação': ['motivado', 'entusiasmado', 'energizado', 'inspirado']
        }
        
        text_lower = text.lower()
        detected_emotions = {}
        
        for emotion, keywords in emotion_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                score = min(1.0, matches / len(keywords) * 2)  # Normalizar
                detected_emotions[emotion] = score
        
        # Análise de sentimento base para contexto
        sentiment_result = self.analyze_sentiment(text)
        
        return {
            'emotions': detected_emotions,
            'primary_emotion': max(detected_emotions.items(), key=lambda x: x[1])[0] if detected_emotions else 'neutro',
            'sentiment': sentiment_result['sentiment'],
            'sentiment_score': sentiment_result['score']
        }
    
    def analyze_aspects(self, text: str) -> Dict:
        """
        Extrai aspectos específicos mencionados no texto.
        
        Args:
            text: Texto para análise
            
        Returns:
            Dicionário com aspectos detectados e sentimentos associados
        """
        text_lower = text.lower()
        aspect_sentiments = {}
        
        for aspect, keywords in self.aspects.items():
            # Verificar se aspecto é mencionado
            mentions = [kw for kw in keywords if kw in text_lower]
            
            if mentions:
                # Extrair sentença relevante
                sentences = text.split('.')
                relevant_sentences = [
                    s for s in sentences
                    if any(kw in s.lower() for kw in mentions)
                ]
                
                if relevant_sentences:
                    # Analisar sentimento do aspecto
                    aspect_text = '. '.join(relevant_sentences)
                    sentiment = self.analyze_sentiment(aspect_text)
                    
                    aspect_sentiments[aspect] = {
                        'mentioned': True,
                        'sentiment': sentiment['sentiment'],
                        'score': sentiment['score'],
                        'keywords_found': mentions
                    }
        
        return {
            'aspects': aspect_sentiments,
            'total_aspects': len(aspect_sentiments)
        }
    
    def analyze_text_collection(self, texts: List[str]) -> Dict:
        """
        Analisa coleção de textos e detecta padrões.
        
        Args:
            texts: Lista de textos para análise
            
        Returns:
            Dicionário com análise agregada
        """
        if not texts:
            return {
                'total_texts': 0,
                'sentiment_distribution': {'positivo': 0.0, 'neutro': 0.0, 'negativo': 0.0},
                'dominant_emotions': {},
                'aspects_analysis': {},
                'overall_sentiment': 'neutro'
            }

        all_sentiments = []
        all_emotions = Counter()
        all_aspects = {}
        
        for text in texts:
            # Sentimento
            sentiment = self.analyze_sentiment(text)
            all_sentiments.append(sentiment['sentiment'])
            
            # Emoções
            emotions = self.analyze_multi_emotion(text)
            for emotion, score in emotions['emotions'].items():
                all_emotions[emotion] += score
            
            # Aspectos
            aspects = self.analyze_aspects(text)
            for aspect, data in aspects['aspects'].items():
                if aspect not in all_aspects:
                    all_aspects[aspect] = {
                        'mentions': 0,
                        'positive': 0,
                        'negative': 0,
                        'neutral': 0
                    }
                
                all_aspects[aspect]['mentions'] += 1
                all_aspects[aspect][data['sentiment']] += 1
        
        # Calcular distribuição de sentimentos
        sentiment_dist = Counter(all_sentiments)
        total = len(texts)
        
        return {
            'total_texts': total,
            'sentiment_distribution': {
                'positivo': sentiment_dist.get('positivo', 0) / total,
                'neutro': sentiment_dist.get('neutro', 0) / total,
                'negativo': sentiment_dist.get('negativo', 0) / total
            },
            'dominant_emotions': dict(all_emotions.most_common(5)),
            'aspects_analysis': all_aspects,
            'overall_sentiment': sentiment_dist.most_common(1)[0][0] if sentiment_dist else 'neutro'
        }
    
    def detect_risk_keywords(self, text: str) -> Dict:
        """
        Detecta palavras-chave indicativas de risco.
        
        Args:
            text: Texto para análise
            
        Returns:
            Dicionário com palavras-chave de risco detectadas
        """
        risk_keywords = {
            'alto_risco': [
                'desistir', 'acabar', 'não aguento', 'cansado demais',
                'sem esperança', 'sem saída', 'sobrecarga extrema'
            ],
            'medio_risco': [
                'sobrecarregado', 'estressado', 'exausto', 'frustrado',
                'sem energia', 'preocupado', 'ansioso'
            ],
            'baixo_risco': [
                'cansado', 'trabalhoso', 'desafiador', 'ocupado'
            ]
        }
        
        text_lower = text.lower()
        detected_risks = {}
        
        for risk_level, keywords in risk_keywords.items():
            found = [kw for kw in keywords if kw in text_lower]
            if found:
                detected_risks[risk_level] = {
                    'keywords_found': found,
                    'count': len(found)
                }
        
        return {
            'risk_keywords': detected_risks,
            'has_high_risk': 'alto_risco' in detected_risks,
            'risk_level': 'alto' if 'alto_risco' in detected_risks else 'medio' if 'medio_risco' in detected_risks else 'baixo'
        }

    def _heuristic_sentiment(self, text: str) -> Dict[str, float]:
        """Fallback simples baseado em palavras-chave."""
        lower_text = text.lower()
        positive_hits = sum(lower_text.count(word) for word in self.positive_keywords)
        negative_hits = sum(lower_text.count(word) for word in self.negative_keywords)
        total_hits = positive_hits + negative_hits

        if total_hits == 0:
            return {
                'sentiment': 'neutro',
                'score': 0.5,
                'confidence': 0.2
            }

        if positive_hits > negative_hits:
            sentiment = 'positivo'
            score = 0.5 + (positive_hits / (total_hits * 2))
        elif negative_hits > positive_hits:
            sentiment = 'negativo'
            score = 0.5 + (negative_hits / (total_hits * 2))
        else:
            sentiment = 'neutro'
            score = 0.5

        confidence = min(1.0, total_hits / 5)

        return {
            'sentiment': sentiment,
            'score': float(min(max(score, 0.0), 1.0)),
            'confidence': float(confidence)
        }
    
    def generate_wordcloud(self, texts: List[str], output_path: str):
        """
        Gera word cloud personalizada de sentimentos predominantes.
        
        Args:
            texts: Lista de textos
            output_path: Caminho para salvar imagem
        """
        if not WORDCLOUD_AVAILABLE:
            logger.warning("wordcloud não disponível. Instale com: pip install wordcloud")
            return
        
        # Combinar todos os textos
        combined_text = ' '.join(texts)
        
        # Gerar word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(combined_text)
        
        # Salvar
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud de Sentimentos', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Word cloud salva em {output_path}")
    
    def track_sentiment_over_time(
        self,
        texts_with_dates: List[Tuple[str, str]]
    ) -> Dict:
        """
        Rastreia mudanças sutis de humor ao longo do tempo.
        
        Args:
            texts_with_dates: Lista de tuplas (texto, data)
            
        Returns:
            Dicionário com análise temporal
        """
        from datetime import datetime
        
        sentiment_timeline = []
        
        for text, date_str in texts_with_dates:
            sentiment = self.analyze_sentiment(text)
            try:
                datetime.fromisoformat(date_str)
            except ValueError:
                logger.warning("Data inválida fornecida: %s", date_str)
                continue
            
            sentiment_timeline.append({
                'date': date_str,
                'sentiment': sentiment['sentiment'],
                'score': sentiment['score']
            })
        
        # Ordenar por data
        sentiment_timeline.sort(key=lambda x: x['date'])
        
        # Detectar tendências
        if len(sentiment_timeline) >= 3:
            recent_scores = [s['score'] for s in sentiment_timeline[-7:]]
            trend = 'melhorando' if recent_scores[-1] > recent_scores[0] else 'piorando' if recent_scores[-1] < recent_scores[0] else 'estável'
        else:
            trend = 'insuficiente_dados'
        
        scores = [s['score'] for s in sentiment_timeline]
        avg_score = float(np.mean(scores)) if scores else 0.0

        return {
            'timeline': sentiment_timeline,
            'trend': trend,
            'average_sentiment': avg_score
        }


if __name__ == "__main__":
    # Exemplo de uso
    analyzer = SentimentAnalyzer()
    
    text = "Estou muito sobrecarregado no trabalho. Os prazos estão impossíveis e não consigo descansar. Estou cansado demais."
    
    # Análise básica
    sentiment = analyzer.analyze_sentiment(text)
    print(f"Sentimento: {sentiment}")
    
    # Análise multi-emoção
    emotions = analyzer.analyze_multi_emotion(text)
    print(f"Emoções: {emotions}")
    
    # Aspectos
    aspects = analyzer.analyze_aspects(text)
    print(f"Aspectos: {aspects}")
    
    # Palavras-chave de risco
    risk = analyzer.detect_risk_keywords(text)
    print(f"Risco: {risk}")

