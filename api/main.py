"""
API FastAPI para servir modelos de IA do WorkWell.
Implementa endpoints para predição, análise e recomendações.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.burnout.lstm_model import BurnoutPredictor
from services.nlp.sentiment_analyzer import SentimentAnalyzer
from services.generative.emotional_support import EmotionalSupportAI
from services.recommendation.recommendation_engine import RecommendationEngine
from models.timeseries.prophet_forecaster import WellbeingForecaster
from vision.fatigue_detector import FatigueDetector

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

security = HTTPBearer()

burnout_predictor = None
sentiment_analyzer = None
emotional_support_ai = None
recommendation_engine = None
wellbeing_forecaster = None
fatigue_detector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global burnout_predictor, sentiment_analyzer, emotional_support_ai
    global recommendation_engine, wellbeing_forecaster, fatigue_detector
    
    logger.info("Inicializando serviços de IA...")
    
    try:
        burnout_predictor = BurnoutPredictor()
        model_path = os.path.join(Path(__file__).parent.parent, "pipelines", "models", "storage", "best_burnout_model.pt")
        if os.path.exists(model_path):
            burnout_predictor.load_model(model_path)
        logger.info("BurnoutPredictor pronto.")
    except Exception as exc:
        logger.exception("Falha ao inicializar BurnoutPredictor: %s", exc)
        burnout_predictor = None

    try:
        sentiment_analyzer = SentimentAnalyzer()
        logger.info("SentimentAnalyzer pronto.")
    except Exception as exc:
        logger.exception("Falha ao inicializar SentimentAnalyzer: %s", exc)
        sentiment_analyzer = None

    try:
        emotional_support_ai = EmotionalSupportAI()
        logger.info("EmotionalSupportAI pronto.")
    except Exception as exc:
        logger.exception("Falha ao inicializar EmotionalSupportAI: %s", exc)
        emotional_support_ai = None

    try:
        recommendation_engine = RecommendationEngine()
        logger.info("RecommendationEngine pronto.")
    except Exception as exc:
        logger.exception("Falha ao inicializar RecommendationEngine: %s", exc)
        recommendation_engine = None

    try:
        wellbeing_forecaster = WellbeingForecaster()
        logger.info("WellbeingForecaster pronto.")
    except Exception as exc:
        logger.exception("Falha ao inicializar WellbeingForecaster: %s", exc)
        wellbeing_forecaster = None

    try:
        fatigue_detector = FatigueDetector()
        logger.info("FatigueDetector pronto.")
    except Exception as exc:
        logger.exception("Falha ao inicializar FatigueDetector: %s", exc)
        fatigue_detector = None
    
    yield
    
    logger.info("Encerrando serviços de IA...")


app = FastAPI(
    title="WorkWell AI API",
    description="API de Inteligência Artificial para prevenção de burnout e bem-estar corporativo",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CheckinData(BaseModel):
    """Dados de check-in para análise."""
    usuario_id: int
    nivel_stress: int = Field(ge=1, le=10)
    horas_trabalhadas: float = Field(ge=0, le=24)
    horas_sono: Optional[float] = Field(None, ge=0, le=24)
    sentimento: Optional[str] = None
    observacoes: Optional[str] = None
    score_bemestar: Optional[float] = Field(None, ge=0, le=100)
    data_checkin: Optional[datetime] = None


class BurnoutPredictionRequest(BaseModel):
    """Request para predição de burnout."""
    usuario_id: int
    checkins: List[CheckinData]
    sequence_length: Optional[int] = 30


class BurnoutPredictionResponse(BaseModel):
    """Response de predição de burnout."""
    usuario_id: int
    predicted_class: str
    probabilities: Dict[str, float]
    risk_score: float
    recommendations: List[str]


class SentimentAnalysisRequest(BaseModel):
    """Request para análise de sentimento."""
    texts: List[str]
    user_id: Optional[int] = None


class SentimentAnalysisResponse(BaseModel):
    """Response de análise de sentimento."""
    results: List[Dict]
    overall_sentiment: str
    risk_keywords: Dict
    dominant_emotions: List[str]


class ChatRequest(BaseModel):
    """Request para chat de suporte emocional."""
    user_id: int
    message: str
    context: Optional[Dict] = None


class ChatResponse(BaseModel):
    """Response do chat."""
    response: str
    sources: List[str]
    confidence: float
    timestamp: str


class RecommendationRequest(BaseModel):
    """Request para recomendações."""
    user_id: int
    user_profile: Dict
    context: Optional[Dict] = None
    n_recommendations: int = 5


class RecommendationResponse(BaseModel):
    """Response de recomendações."""
    recommendations: List[Dict]
    user_id: int


class ForecastHistoryEntry(BaseModel):
    date: datetime
    score: float


class ForecastRequest(BaseModel):
    """Request para previsão de séries temporais."""
    user_id: int
    periods: int = 30
    target_column: str = "score_bemestar"
    history: Optional[List[ForecastHistoryEntry]] = None


class ForecastResponse(BaseModel):
    """Response de previsão."""
    forecast: List[Dict]
    risk_periods: List[Dict]
    seasonality_analysis: Dict


class FeedbackRequest(BaseModel):
    """Request para feedback de recomendações."""
    user_id: int
    item_id: str
    rating: float = Field(ge=0, le=5)
    completed: bool = True


@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "online",
        "service": "WorkWell AI API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check detalhado."""
    return {
        "status": "healthy",
        "services": {
            "burnout_predictor": burnout_predictor is not None,
            "sentiment_analyzer": sentiment_analyzer is not None,
            "emotional_support": emotional_support_ai is not None,
            "recommendation_engine": recommendation_engine is not None,
            "wellbeing_forecaster": wellbeing_forecaster is not None,
            "fatigue_detector": fatigue_detector is not None
        }
    }


@app.post("/api/v1/burnout/predict", response_model=BurnoutPredictionResponse)
async def predict_burnout(request: BurnoutPredictionRequest):
    """
    Prediz risco de burnout usando modelo LSTM.
    """
    if burnout_predictor is None:
        raise HTTPException(status_code=503, detail="Serviço de predição não disponível")
    
    try:
        import pandas as pd
        df = pd.DataFrame([c.dict() for c in request.checkins])

        sequence_length = request.sequence_length or 30
        X, _ = burnout_predictor.prepare_sequences(df, sequence_length=sequence_length)

        if len(X) == 0:
            raise HTTPException(status_code=400, detail="Dados insuficientes para predição")
        
        prediction = burnout_predictor.predict(X[-1])
        
        num_classes = len(prediction['probabilities'])
        max_prob = max(prediction['probabilities'].values())
        
        if num_classes == 1 or max_prob < 0.3:
            last_checkin_raw = request.checkins[-1]
            
            stress = float(last_checkin_raw.nivel_stress)
            horas_trabalhadas = float(last_checkin_raw.horas_trabalhadas)
            horas_sono = float(last_checkin_raw.horas_sono or 7.0)
            score_bemestar = float(last_checkin_raw.score_bemestar or 60.0)
            
            stress_component = (stress / 10.0) * 40
            work_component = min(horas_trabalhadas / 16.0, 1.0) * 20
            sleep_component = max(0.0, (8 - horas_sono) / 8.0) * 20
            wellbeing_component = max(0.0, (100 - score_bemestar) / 100.0) * 20
            
            risk_score_total = stress_component + work_component + sleep_component + wellbeing_component
            
            if risk_score_total < 25:
                risk_class = "baixo"
            elif risk_score_total < 50:
                risk_class = "medio"
            elif risk_score_total < 75:
                risk_class = "alto"
            else:
                risk_class = "critico"
            
            risk_score = risk_score_total / 100.0
            
            risk_map = {"baixo": 0.0, "medio": 0.33, "alto": 0.66, "critico": 1.0}
            probabilities = {
                "baixo": 0.0,
                "medio": 0.0,
                "alto": 0.0,
                "critico": 0.0
            }
            probabilities[risk_class] = 1.0
            
            prediction = {
                "predicted_class": risk_class,
                "probabilities": probabilities,
                "risk_score": risk_score
            }
        
        recommendations = []
        if prediction['predicted_class'] in ['alto', 'critico']:
            recommendations = [
                "Considere reduzir horas de trabalho",
                "Pratique técnicas de mindfulness",
                "Busque apoio profissional se necessário"
            ]
        elif prediction['predicted_class'] == 'medio':
            recommendations = [
                "Monitore seus níveis de stress",
                "Mantenha um equilíbrio entre trabalho e descanso"
            ]
        
        return BurnoutPredictionResponse(
            usuario_id=request.usuario_id,
            predicted_class=prediction['predicted_class'],
            probabilities=prediction['probabilities'],
            risk_score=prediction['risk_score'],
            recommendations=recommendations
        )
    except ValueError as ve:
        logger.error(f"Erro de validação na predição: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sentiment/analyze", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """
    Analisa sentimento de textos usando BERT.
    """
    if sentiment_analyzer is None:
        raise HTTPException(status_code=503, detail="Serviço de análise não disponível")
    
    try:
        results = []
        all_emotions = []
        risk_keywords_combined = {}
        
        for text in request.texts:
            sentiment = sentiment_analyzer.analyze_sentiment(text)
            
            emotions = sentiment_analyzer.analyze_multi_emotion(text)
            all_emotions.extend(list(emotions['emotions'].keys()))
            
            risk = sentiment_analyzer.detect_risk_keywords(text)
            for risk_level, data in risk['risk_keywords'].items():
                if risk_level not in risk_keywords_combined:
                    risk_keywords_combined[risk_level] = []
                risk_keywords_combined[risk_level].extend(data['keywords_found'])
            
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "sentiment": sentiment['sentiment'],
                "score": sentiment['score'],
                "emotions": emotions['emotions'],
                "risk_level": risk['risk_level']
            })
        
        from collections import Counter
        sentiment_counts = Counter([r['sentiment'] for r in results])
        overall_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else 'neutro'
        emotion_counts = Counter(all_emotions)
        dominant_emotions = [emotion for emotion, _ in emotion_counts.most_common(5)]
        
        return SentimentAnalysisResponse(
            results=results,
            overall_sentiment=overall_sentiment,
            risk_keywords=risk_keywords_combined,
            dominant_emotions=dominant_emotions
        )
    except Exception as e:
        logger.error(f"Erro na análise: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat/support", response_model=ChatResponse)
async def emotional_support_chat(request: ChatRequest):
    """
    Chat de suporte emocional usando IA generativa.
    """
    if emotional_support_ai is None:
        raise HTTPException(status_code=503, detail="Serviço de chat não disponível")
    
    try:
        response = emotional_support_ai.chat(
            user_id=request.user_id,
            message=request.message,
            context=request.context
        )
        
        return ChatResponse(**response)
    except Exception as e:
        logger.error(f"Erro no chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Obtém recomendações personalizadas usando engine híbrida.
    """
    if recommendation_engine is None:
        raise HTTPException(status_code=503, detail="Serviço de recomendações não disponível")
    
    try:
        recommendations = recommendation_engine.hybrid_recommend(
            user_id=request.user_id,
            user_profile=request.user_profile,
            context=request.context,
            n_recommendations=request.n_recommendations
        )
        
        return RecommendationResponse(
            recommendations=[
                {
                    "item_id": rec.item_id,
                    "item_type": rec.item_type,
                    "title": rec.title,
                    "description": rec.description,
                    "score": rec.score,
                    "reason": rec.reason
                }
                for rec in recommendations
            ],
            user_id=request.user_id
        )
    except Exception as e:
        logger.error(f"Erro nas recomendações: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/forecast/wellbeing", response_model=ForecastResponse)
async def forecast_wellbeing(request: ForecastRequest):
    """
    Preve tendências futuras de bem-estar usando Prophet.
    """
    if wellbeing_forecaster is None:
        raise HTTPException(status_code=503, detail="Serviço de previsão não disponível")
    
    try:
        import pandas as pd

        if request.history:
            df = pd.DataFrame([
                {
                    "usuario_id": request.user_id,
                    "data_checkin": entry.date,
                    request.target_column: entry.score,
                }
                for entry in request.history
            ])
            wellbeing_forecaster.prepare_data(df, request.user_id, target_column=request.target_column)
            wellbeing_forecaster.train_model(request.user_id)
        elif request.user_id not in wellbeing_forecaster.models:
            raise HTTPException(status_code=400, detail="Forneça histórico inicial em 'history' para treinar o previsor.")

        forecast = wellbeing_forecaster.forecast(
            user_id=request.user_id,
            periods=request.periods
        )
        
        risk_periods = wellbeing_forecaster.predict_high_risk_periods(
            user_id=request.user_id,
            forecast_days=request.periods
        )
        
        seasonality = wellbeing_forecaster.get_seasonality_analysis(request.user_id)
        
        return ForecastResponse(
            forecast=[
                {
                    "date": row['ds'].isoformat(),
                    "predicted": float(row['yhat']),
                    "lower": float(row['yhat_lower']),
                    "upper": float(row['yhat_upper'])
                }
                for _, row in forecast.iterrows()
            ],
            risk_periods=risk_periods,
            seasonality_analysis=seasonality
        )
    except ValueError as ve:
        logger.error(f"Erro na previsão (dados): {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Erro na previsão: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submete feedback para melhorar recomendações.
    """
    if recommendation_engine is None:
        raise HTTPException(status_code=503, detail="Serviço não disponível")
    
    try:
        recommendation_engine.update_feedback(
            user_id=request.user_id,
            item_id=request.item_id,
            rating=request.rating,
            completed=request.completed,
        )
        return {"status": "success", "message": "Feedback registrado"}
    except Exception as e:
        logger.error(f"Erro ao registrar feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("API_RELOAD", "true").lower() == "true"
    )
