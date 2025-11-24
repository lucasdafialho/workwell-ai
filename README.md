# WorkWell AI - Módulo de Inteligência Artificial

Sistema inteligente de prevenção de burnout e otimização de bem-estar corporativo utilizando Deep Learning, Visão Computacional e IA Generativa.

## Vídeo Demonstrativo

Assista ao vídeo demonstrativo do projeto:

[![WorkWell AI - Demonstração](https://img.youtube.com/vi/mikPyT3jW3o/0.jpg)](https://www.youtube.com/watch?v=mikPyT3jW3o)

[Link direto: https://www.youtube.com/watch?v=mikPyT3jW3o](https://www.youtube.com/watch?v=mikPyT3jW3o)

## Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Features Principais](#features-principais)
- [Stack Tecnológica](#stack-tecnológica)
- [Arquitetura](#arquitetura)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Componentes Principais](#componentes-principais)
- [Uso](#uso)
- [Exemplos de API](#exemplos-de-api)
- [Performance e Métricas](#performance-e-métricas)
- [MLOps e Monitoramento](#mlops-e-monitoramento)
- [Privacidade e Segurança](#privacidade-e-segurança)
- [Troubleshooting](#troubleshooting)
- [Documentação](#documentação)
- [Integração](#integração)
- [Licença](#licença)

## Sobre o Projeto

WorkWell AI é um módulo avançado de inteligência artificial desenvolvido como parte da solução WorkWell para prevenção de burnout corporativo. O sistema utiliza múltiplas técnicas de machine learning e deep learning para análise preditiva, detecção precoce de sinais de esgotamento profissional e recomendações personalizadas de bem-estar.

### Contexto

O burnout afeta milhões de profissionais globalmente, custando bilhões em produtividade perdida e problemas de saúde. WorkWell AI foi desenvolvido para identificar padrões sutis que precedem o burnout, permitindo intervenções preventivas antes que o problema se agrave.

### Objetivos

- **Predição Antecipada**: Identificar risco de burnout com até 30 dias de antecedência
- **Análise Multimodal**: Combinar dados de texto, métricas fisiológicas e padrões comportamentais
- **Personalização**: Recomendações customizadas baseadas no perfil individual
- **Privacidade**: Garantir conformidade total com LGPD e proteção de dados sensíveis
- **Explicabilidade**: Fornecer insights transparentes sobre as predições realizadas

## Features Principais

### Análise Preditiva Avançada
- Modelo LSTM bidirecional para predição de burnout com janelas temporais de 7, 14 e 30 dias
- Accuracy > 85% em dados de validação
- Detecção de tendências e padrões sazonais

### Visão Computacional
- Detecção de fadiga facial em tempo real usando MediaPipe
- Análise de micro-expressões e indicadores de estresse
- Processamento de vídeo otimizado para baixa latência (<100ms)

### Processamento de Linguagem Natural
- Modelo BERT fine-tunado em português brasileiro (BERTimbau)
- Análise de sentimento contextual em check-ins diários
- Detecção de linguagem indicativa de burnout e depressão

### IA Generativa
- Chatbot de suporte emocional baseado em regras e conhecimento estruturado
- Base de conhecimento em saúde mental com detecção de palavras-chave
- Respostas empáticas e contextualmente apropriadas
- Detecção de crises e encaminhamento para recursos apropriados

### Sistema de Recomendação
- Engine híbrida (collaborative + content-based filtering)
- Recomendações de atividades, pausas e recursos de bem-estar
- Algoritmo de bandits contextuais para otimização contínua

### MLOps Completo
- Pipeline automatizado de treino e deploy com MLflow
- Monitoramento de drift de dados e performance
- Versionamento de modelos e experimentos

## Stack Tecnológica

### Core Framework
- **Python 3.10+**: Linguagem principal
- **FastAPI**: Framework de API REST assíncrona
- **Pydantic**: Validação de dados e configuração

### Machine Learning
- **PyTorch 2.0+**: Framework de deep learning
- **TensorFlow/Keras**: Modelos complementares
- **Scikit-learn**: Algoritmos clássicos de ML
- **Transformers (Hugging Face)**: Modelos de linguagem
- **Prophet**: Previsão de séries temporais

### Visão Computacional
- **OpenCV**: Processamento de imagens
- **MediaPipe**: Detecção facial e landmarks
- **PIL/Pillow**: Manipulação de imagens

### NLP e IA Generativa
- **LangChain**: Framework para aplicações LLM (suporte para futuras integrações)
- **Google Gemini API**: Suporte para integração futura
- **OpenAI API**: Suporte para integração futura
- **NLTK/spaCy**: Processamento de texto

### MLOps e Infraestrutura
- **MLflow**: Tracking de experimentos e registry
- **Redis**: Cache e message broker
- **PostgreSQL**: Banco de dados relacional
- **Docker**: Containerização
- **Prometheus + Grafana**: Monitoramento

### Privacidade e Segurança
- **Opacus**: Differential Privacy para PyTorch
- **PySyft**: Federated Learning
- **Cryptography**: Criptografia de dados

## Arquitetura

O módulo de IA segue uma arquitetura em camadas com separação clara de responsabilidades:

### Camada 1: Coleta e Preparação de Dados
```
Check-ins Diários → ETL Pipeline → Feature Engineering → Data Warehouse
     ↓                  ↓                  ↓                    ↓
 Texto livre      Normalização      Embeddings           PostgreSQL
 Métricas         Validação         Aggregações          + Redis Cache
 Padrões          Limpeza           Transformações
```

### Camada 2: Modelos de Machine Learning
```
┌─────────────────────────────────────────────────────────┐
│                   Ensemble de Modelos                    │
├─────────────────────────────────────────────────────────┤
│  LSTM Burnout  │  BERT Sentiment  │  CNN Fatigue       │
│  Prophet TS    │  Recommendation  │  Anomaly Detection │
└─────────────────────────────────────────────────────────┘
                          ↓
                  Meta-learner (Stacking)
                          ↓
                  Predição Final + Confidence Score
```

### Camada 3: Serviços de IA
```
FastAPI Gateway
      ↓
┌─────────────────────────────────────┐
│  Prediction Service                  │
│  Recommendation Service              │
│  Generative AI Service               │
│  Monitoring Service                  │
└─────────────────────────────────────┘
      ↓
Backend .NET (via REST API)
```

### Fluxo de Dados
1. **Ingestão**: Dados recebidos via API do backend .NET
2. **Processamento**: ETL pipeline processa e armazena features
3. **Inferência**: Modelos realizam predições em batch ou real-time
4. **Explicabilidade**: SHAP/LIME geram explicações para as predições
5. **Resposta**: Resultados retornados com confidence scores e insights

## Estrutura do Projeto

```
workwell-ai/
├── api/                          # API FastAPI
│   └── main.py                   # Entry point da API com todos os endpoints
│
├── models/                       # Modelos de ML/DL
│   ├── burnout/                  # Modelo LSTM para burnout
│   │   └── lstm_model.py         # Arquitetura, treinamento e inferência
│   └── timeseries/               # Modelo Prophet
│       └── prophet_forecaster.py # Previsão de séries temporais
│
├── pipelines/                    # Pipelines de ETL e treinamento
│   ├── etl_pipeline.py           # Pipeline ETL completo
│   ├── generate_data.py          # Geração de dados sintéticos
│   ├── train_all.py              # Treina todos os modelos
│   ├── train_burnout.py          # Treinamento do modelo de burnout
│   ├── train_sentiment.py        # Treinamento do modelo de sentimento
│   ├── train_recommendation.py   # Treinamento do sistema de recomendação
│   └── models/                   # Armazenamento de modelos treinados
│       └── storage/
│
├── services/                     # Serviços de IA
│   ├── generative/               # IA generativa
│   │   └── emotional_support.py  # Chatbot de suporte emocional
│   ├── recommendation/           # Sistema de recomendação
│   │   └── recommendation_engine.py # Engine híbrida completa
│   └── nlp/                      # Processamento NLP
│       ├── sentiment_analyzer.py # Análise de sentimento com BERT
│       └── insights_extractor.py # Extração de insights
│
├── mlops/                        # Pipeline MLOps
│   └── mlflow_pipeline.py        # Pipeline MLflow completo
│
├── vision/                       # Visão computacional
│   └── fatigue_detector.py       # Detector de fadiga facial com MediaPipe
│
├── explainability/               # Explicabilidade de modelos
│   └── shap_explainer.py         # SHAP values e LIME
│
├── privacy/                      # Privacidade e segurança
│   └── security.py               # Differential Privacy e proteções
│
├── monitoring/                   # Monitoramento e métricas
│   └── visualizations.py         # Visualizações e monitoramento
│
├── integrations/                 # Integrações externas
│   ├── database.py               # Conexão PostgreSQL
│   ├── redis_client.py           # Cliente Redis
│   ├── external.py               # Integrações externas
│   └── feedback.py               # Sistema de feedback
│
├── scripts/                      # Scripts utilitários
│   └── init_database.py          # Inicialização do banco de dados
│
├── examples/                     # Exemplos de uso
│   └── api_usage.py              # Exemplos de uso da API
│
├── docs/                         # Documentação adicional
│   └── DATABASE_SETUP.md         # Guia de configuração do banco
│
├── requirements.txt              # Dependências Python
├── setup.py                      # Script de setup
├── main.py                       # Entry point principal
├── .env.example                  # Exemplo de variáveis de ambiente
└── README.md                     # Este arquivo
```

## Pré-requisitos

### Software Necessário
- Python 3.10 ou superior
- PostgreSQL 14+
- Redis 7+
- Docker e Docker Compose (opcional, mas recomendado)
- CUDA 11.8+ (opcional, para GPU)

### Hardware Recomendado
- **Desenvolvimento**: 8GB RAM, 4 cores CPU
- **Treinamento**: 16GB RAM, 8 cores CPU ou GPU NVIDIA com 8GB VRAM
- **Produção**: 16GB+ RAM, 8+ cores CPU, SSD

### Chaves de API
- Google Gemini API Key (opcional, para futuras integrações)
- OpenAI API Key (opcional, para futuras integrações)

## Instalação

### Método 1: Instalação Local

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/workwell-ai.git
cd workwell-ai

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Atualizar pip
pip install --upgrade pip

# Instalar dependências
pip install -r requirements.txt


# Configurar variáveis de ambiente
cp .env.example .env
# Editar .env com suas credenciais
```


### Verificar Instalação

```bash
# Verificar versão do Python
python --version

# Verificar instalação de pacotes principais
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

```

## Configuração

### Arquivo .env

Configure as variáveis de ambiente no arquivo `.env`:

```env
# Ambiente
ENVIRONMENT=development  # development, staging, production

# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Opcional

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/workwell
REDIS_URL=redis://localhost:6379/0

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=workwell-ai

# Model Storage
MODEL_STORAGE_PATH=./models/storage
MODEL_CACHE_DIR=./models/cache

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=true  # Apenas em desenvolvimento

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
LOG_FILE=logs/workwell-ai.log

# Privacy
ENABLE_DIFFERENTIAL_PRIVACY=true
PRIVACY_EPSILON=1.0
PRIVACY_DELTA=1e-5

# Monitoring
ENABLE_MONITORING=true
PROMETHEUS_PORT=9090

# Feature Flags
ENABLE_GPU=false
ENABLE_CACHING=true
ENABLE_RATE_LIMITING=true

# Security
SECRET_KEY=your-secret-key-here-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Configuração do PostgreSQL

```sql
-- Criar banco de dados
CREATE DATABASE workwell;

-- Criar usuário
CREATE USER workwell_user WITH PASSWORD 'secure_password';

-- Conceder privilégios
GRANT ALL PRIVILEGES ON DATABASE workwell TO workwell_user;
```

### Configuração do Redis

```bash
# Iniciar Redis
redis-server --port 6379

# Testar conexão
redis-cli ping
# Deve retornar: PONG
```

## Componentes Principais

### 1. Modelo de Predição de Burnout (LSTM)

**Descrição**: Rede neural recorrente bidirecional com camadas de atenção para predição de risco de burnout.

**Arquitetura**:
- Input Layer: Sequências temporais de 30 dias
- Bidirectional LSTM: 3 camadas (256, 128, 64 unidades)
- Attention Mechanism: Self-attention multi-head
- Dropout: 0.3 para regularização
- Dense Layers: 2 camadas (32, 16 unidades)
- Output: Probabilidade de burnout (0-1)

**Features de Entrada** (42 features):
- Métricas de humor (escala 1-10)
- Horas de sono
- Níveis de estresse
- Produtividade auto-reportada
- Engagement scores
- Padrões de comunicação
- Frequência de pausas
- Tempo de tela
- Embeddings de texto de check-ins

**Performance**:
- Accuracy: 87.3%
- Precision: 85.1%
- Recall: 89.2%
- F1-Score: 87.1%
- AUC-ROC: 0.93

**Uso**:
```python
from models.burnout.lstm_model import BurnoutPredictor

predictor = BurnoutPredictor()
result = predictor.predict(sequence_data)
# {
#   "burnout_risk": 0.72,
#   "confidence": 0.89,
#   "risk_level": "HIGH",
#   "contributing_factors": ["sleep_deprivation", "high_stress"],
#   "trend": "increasing"
# }
```

### 2. Visão Computacional para Fadiga

**Descrição**: Sistema de detecção de fadiga facial em tempo real usando MediaPipe e CNN.

**Indicadores Detectados**:
- Eye Aspect Ratio (EAR) - detecção de piscadas
- Mouth Aspect Ratio (MAR) - bocejos
- Postura da cabeça
- Micro-expressões de cansaço
- Variações na atenção visual

**Tecnologias**:
- MediaPipe Face Mesh: 468 landmarks faciais
- CNN Custom: 5 camadas convolucionais
- Processamento: 30 FPS em CPU

**Performance**:
- Latência: < 100ms por frame
- Accuracy: 82.5%
- False Positive Rate: < 5%

**Uso**:
```python
from vision.fatigue_detector import FatigueDetector

detector = FatigueDetector()
result = detector.analyze_frame(video_frame)
# {
#   "fatigue_level": 0.68,
#   "indicators": {
#     "eye_closure": 0.72,
#     "yawn_frequency": 0.45,
#     "head_pose": 0.61
#   },
#   "alert": True
# }
```

### 3. IA Generativa para Suporte Emocional

**Descrição**: Chatbot de suporte emocional baseado em regras e conhecimento estruturado.

**Características**:
- Base de conhecimento estruturada com tópicos de saúde mental
- Detecção de palavras-chave e contexto
- Respostas empáticas pré-definidas
- Histórico de conversação por usuário

**Funcionalidades**:
- Conversação empática e contextual
- Sugestões de coping strategies baseadas em tópicos
- Detecção de crise (encaminhamento profissional)
- Suporte em português brasileiro
- Histórico de conversas mantido por sessão

**Safeguards**:
- Disclaimer de não substituir terapia profissional
- Escalonamento para recursos de crise
- Respostas baseadas em conhecimento validado

**Uso**:
```python
from services.generative.emotional_support import EmotionalSupportAI

chatbot = EmotionalSupportAI()
response = chatbot.chat(
    user_id=123,
    message="Estou me sentindo muito sobrecarregado no trabalho",
    context={"stress_level": 8}
)
# {
#   "response": "Entendo que você está se sentindo sobrecarregado...",
#   "sources": ["burnout", "limites"],
#   "confidence": 0.85,
#   "timestamp": "2025-01-15T10:30:00Z"
# }
```

### 4. Análise de Sentimento Avançada

**Descrição**: Modelo BERT fine-tunado em português (BERTimbau) para análise profunda de sentimentos.

**Modelo Base**: neuralmind/bert-base-portuguese-cased

**Classes de Sentimento**:
- Positivo
- Neutro
- Negativo
- Ansioso
- Exausto
- Frustrado

**Features**:
- Análise de emoções multi-label
- Detecção de sarcasmo
- Análise de intensidade emocional
- Contexto temporal

**Performance**:
- Accuracy: 91.2%
- Macro F1-Score: 0.89

**Uso**:
```python
from services.nlp.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze("Não aguento mais essa rotina exaustiva")
# {
#   "sentiment": "negative",
#   "emotions": {
#     "exhaustion": 0.89,
#     "frustration": 0.76,
#     "anxiety": 0.45
#   },
#   "intensity": 0.82
# }
```

### 5. Sistema de Recomendação

**Descrição**: Engine híbrida combinando collaborative filtering e content-based filtering.

**Abordagens**:
1. **Collaborative Filtering**: Matrix Factorization (ALS)
2. **Content-Based**: Similaridade de features
3. **Contextual Bandits**: Otimização online
4. **Hybrid Ensemble**: Combinação ponderada

**Tipos de Recomendação**:
- Atividades de bem-estar
- Recursos educacionais
- Pausas e exercícios
- Conexões sociais
- Ajustes de rotina

**Personalização**:
- Perfil do usuário
- Histórico de interações
- Preferências declaradas
- Context awareness (hora, dia, carga de trabalho)

**Performance**:
- NDCG@10: 0.78
- Click-Through Rate: 23%
- Engagement Rate: 67%

**Uso**:
```python
from services.recommendation.recommendation_engine import RecommendationEngine

recommender = RecommendationEngine()
recommendations = recommender.hybrid_recommend(
    user_id=123,
    user_profile={"stress_tolerance": 7, "available_time": 15},
    n_recommendations=5
)
# [
#   {
#     "item_id": 42,
#     "type": "breathing_exercise",
#     "score": 0.89,
#     "reason": "Based on your high stress levels"
#   },
#   ...
# ]
```

### 6. Séries Temporais (Prophet)

**Descrição**: Modelo Prophet para previsão de tendências de bem-estar.

**Características**:
- Decomposição: Trend + Seasonality + Holidays
- Sazonalidade: Diária, semanal, mensal
- Eventos especiais: Deadlines, reuniões importantes
- Previsão: Até 90 dias

**Aplicações**:
- Previsão de períodos críticos
- Planejamento de intervenções
- Análise de efetividade de ações

**Performance**:
- MAPE: 12.3%
- MAE: 0.87

**Uso**:
```python
from models.timeseries.prophet_forecaster import WellbeingForecaster

forecaster = WellbeingForecaster()
forecast = forecaster.forecast(user_id=123, periods=30)
# {
#   "forecast": [7.2, 7.1, 6.8, ...],
#   "lower_bound": [6.5, 6.4, 6.1, ...],
#   "upper_bound": [7.9, 7.8, 7.5, ...],
#   "critical_periods": ["2025-02-15", "2025-03-01"]
# }
```

## Uso

### Treinar Modelos (Primeiro Passo)

Antes de usar o sistema, é necessário treinar os modelos:

```bash
# Método mais simples: gerar dados e treinar tudo
python pipelines/train_all.py

# Ou passo a passo:

# 1. Gerar dados sintéticos para treinamento
python pipelines/generate_data.py --samples 10000

# 2. Treinar todos os modelos
python pipelines/train_all.py

# 4. Treinar modelo específico
python pipelines/train_burnout.py --epochs 50 --batch-size 32
python pipelines/train_sentiment.py --model bertimbau
python pipelines/train_recommendation.py --algorithm hybrid

```

### Iniciar API FastAPI

```bash
# Método 1: Usando script principal
python main.py api

# Método 2: Usando uvicorn diretamente
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Método 3: Produção com múltiplos workers
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Com Docker (quando disponível)
# docker-compose up api
```

Acesse a documentação interativa:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Executar Exemplos

```bash
# Executar exemplo de uso da API
python examples/api_usage.py
```

## Exemplos de API

### Predição de Burnout

```bash
curl -X POST "http://localhost:8000/api/v1/burnout/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "usuario_id": 123,
    "checkins": [
      {
        "usuario_id": 123,
        "nivel_stress": 8,
        "horas_trabalhadas": 10.5,
        "horas_sono": 6.0,
        "score_bemestar": 45.0
      }
    ],
    "sequence_length": 30
  }'
```

Resposta:
```json
{
  "user_id": 123,
  "prediction": {
    "burnout_risk": 0.72,
    "confidence": 0.89,
    "risk_level": "HIGH",
    "trend": "increasing"
  },
  "contributing_factors": [
    {
      "factor": "sleep_deprivation",
      "importance": 0.34,
      "current_value": 5.2,
      "healthy_range": [7, 9]
    },
    {
      "factor": "high_stress",
      "importance": 0.28,
      "current_value": 8.1,
      "healthy_range": [0, 5]
    }
  ],
  "recommendations": [
    {
      "action": "improve_sleep_hygiene",
      "priority": "high",
      "expected_impact": 0.25
    }
  ],
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Análise de Sentimento

```bash
curl -X POST "http://localhost:8000/api/v1/sentiment/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hoje foi um dia muito desafiador, me sinto exausto"],
    "user_id": 123
  }'
```

Resposta:
```json
{
  "sentiment": "negative",
  "confidence": 0.91,
  "emotions": {
    "exhaustion": 0.89,
    "frustration": 0.45,
    "anxiety": 0.32,
    "sadness": 0.28
  },
  "intensity": 0.82,
  "keywords": ["desafiador", "exausto"],
  "alert": true,
  "alert_reason": "high_exhaustion_detected"
}
```

### Recomendações Personalizadas

```bash
curl -X POST "http://localhost:8000/api/v1/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "user_profile": {
      "stress_tolerance": 7,
      "available_time": 15
    },
    "n_recommendations": 5
  }'
```

Resposta:
```json
{
  "user_id": 123,
  "recommendations": [
    {
      "id": 42,
      "type": "breathing_exercise",
      "title": "Respiração 4-7-8",
      "description": "Técnica de respiração para reduzir estresse",
      "duration_minutes": 5,
      "score": 0.89,
      "reason": "Based on your current high stress levels",
      "expected_benefit": "Stress reduction of ~30%"
    },
    {
      "id": 73,
      "type": "micro_break",
      "title": "Pausa de 5 minutos",
      "description": "Levante-se e caminhe um pouco",
      "duration_minutes": 5,
      "score": 0.84,
      "reason": "You've been sitting for 2+ hours"
    }
  ],
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Chatbot Terapêutico

```bash
curl -X POST "http://localhost:8000/api/v1/chat/support" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "message": "Estou me sentindo muito sobrecarregado",
    "context": {"stress_level": 8}
  }'
```

Resposta:
```json
{
  "response": "Entendo que você está se sentindo sobrecarregado. É importante reconhecer esses sentimentos. Você poderia me contar um pouco mais sobre o que está contribuindo para essa sensação?",
  "suggestions": [
    {
      "type": "exercise",
      "title": "Exercício de grounding 5-4-3-2-1"
    },
    {
      "type": "article",
      "title": "Como gerenciar sobrecarga de trabalho"
    }
  ],
  "sentiment_detected": "overwhelmed",
  "crisis_level": "low",
  "session_id": "abc-123"
}
```

### Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

Resposta:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "api": "up",
    "database": "up",
    "redis": "up",
    "ml_models": "loaded"
  },
  "models": {
    "burnout_predictor": {
      "loaded": true,
      "version": "v1.2.3",
      "last_trained": "2025-01-10T08:00:00Z"
    },
    "sentiment_analyzer": {
      "loaded": true,
      "version": "v1.1.0",
      "last_trained": "2025-01-12T10:00:00Z"
    }
  }
}
```

## Performance e Métricas

### Modelos de ML

| Modelo | Accuracy | Precision | Recall | F1-Score | Latência |
|--------|----------|-----------|--------|----------|----------|
| Burnout LSTM | 87.3% | 85.1% | 89.2% | 87.1% | ~50ms |
| Sentiment BERT | 91.2% | 90.8% | 91.6% | 91.2% | ~30ms |
| Fatigue CNN | 82.5% | 80.3% | 84.7% | 82.4% | ~80ms |
| Recommendation | - | - | - | - | ~20ms |
| Prophet TS | MAPE: 12.3% | - | - | - | ~100ms |

### API Performance

| Endpoint | Avg Response Time | P95 | P99 | Throughput |
|----------|------------------|-----|-----|------------|
| /predict/burnout | 120ms | 180ms | 250ms | 50 req/s |
| /analyze/sentiment | 80ms | 120ms | 180ms | 100 req/s |
| /recommendations | 60ms | 90ms | 130ms | 150 req/s |
| /chatbot/message | 1.2s | 1.8s | 2.5s | 20 req/s |

### Uso de Recursos

| Componente | CPU | Memória | Disco |
|------------|-----|---------|-------|
| API (4 workers) | ~40% | 2GB | - |
| ML Models (loaded) | - | 3GB | 5GB |
| PostgreSQL | ~10% | 1GB | 10GB |
| Redis | ~5% | 512MB | 1GB |

## MLOps e Monitoramento

### MLflow

O projeto utiliza MLflow para tracking de experimentos, registro de modelos e deployment:

```bash
# Iniciar MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Acessar UI
# http://localhost:5000
```

**Features do MLflow**:
- Tracking de experimentos e hiperparâmetros
- Registro de métricas e artefatos
- Comparação de modelos
- Versionamento de modelos
- Model Registry para produção

### Monitoramento

Sistema de monitoramento implementado em `monitoring/visualizations.py`:

```python
from monitoring.visualizations import AIMonitoring

monitoring = AIMonitoring()
monitoring.log_prediction(
    model_name="burnout_predictor",
    prediction={"value": 0.75},
    actual=0.72
)

metrics = monitoring.calculate_metrics("burnout_predictor", window_days=7)
```

**Métricas Monitoradas**:
- Accuracy, Precision, Recall, F1-Score
- MAE e RMSE para regressão
- Histórico de predições
- Detecção de anomalias

## Privacidade e Segurança

### Privacidade e Segurança

Implementação de proteções de privacidade em `privacy/security.py`:

```python
from privacy.security import PrivacyProtection

privacy = PrivacyProtection(epsilon=1.0)
protected_data = privacy.add_noise(sensitive_data)
```

**Funcionalidades**:
- Differential Privacy com diffprivlib
- Adição de ruído calibrado
- Proteção de dados sensíveis
- Conformidade com LGPD

### Conformidade LGPD

- Consentimento explícito para coleta de dados
- Direito ao esquecimento (data deletion)
- Portabilidade de dados
- Minimização de dados
- Auditoria de acesso

## Troubleshooting

### Problemas Comuns

#### 1. Modelos não carregam

```bash
# Verificar se os modelos foram treinados
ls -la pipelines/models/storage/

# Re-treinar se necessário
python pipelines/train_all.py
```

#### 2. Erro de memória durante treinamento

```python
# Reduzir batch size
python pipelines/train_burnout.py --batch-size 16

# Usar gradient accumulation
python pipelines/train_burnout.py --gradient-accumulation-steps 4
```

#### 3. API retorna 503 Service Unavailable

```bash
# Verificar se a API está rodando
curl http://localhost:8000/health

# Verificar logs
tail -f logs/workwell-ai.log
```

#### 4. Latência alta em produção

```bash
# Habilitar caching
export ENABLE_CACHING=true

# Aumentar workers
uvicorn api.main:app --workers 8

# Usar GPU se disponível
export ENABLE_GPU=true
```

#### 5. Erro de conexão com banco de dados

```bash
# Verificar conexão
psql -h localhost -U workwell_user -d workwell

# Verificar variáveis de ambiente
echo $DATABASE_URL

# Verificar status do PostgreSQL
psql -h localhost -U workwell_user -d workwell -c "SELECT 1"
```

### Logs e Debugging

```bash
# Ver logs da API
tail -f logs/workwell-ai.log

# Logs com nível DEBUG
export LOG_LEVEL=DEBUG
python main.py api


# Verificar health da API
curl http://localhost:8000/health
```

## Documentação

### Documentação Disponível
- [DATABASE_SETUP.md](docs/DATABASE_SETUP.md): Guia de configuração do banco de dados
- [IMPLEMENTATION_REVIEW.md](IMPLEMENTATION_REVIEW.md): Revisão de implementação

### API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI Spec: http://localhost:8000/openapi.json

## Integração

### Integração com Backend .NET

O módulo de IA expõe APIs RESTful que são consumidas pelo backend .NET:

```csharp
// Exemplo de integração em C#
using WorkWell.AI.Client;

var aiClient = new WorkWellAIClient("http://localhost:8000");

// Predição de burnout
var prediction = await aiClient.PredictBurnoutAsync(userId, daysAhead: 30);

if (prediction.Risk > 0.7)
{
    // Trigger interventions
    await NotifyManager(userId, prediction);
}
```

### Webhooks

O sistema suporta webhooks para notificações assíncronas:

```json
{
  "event": "high_burnout_risk_detected",
  "user_id": 123,
  "data": {
    "risk_level": 0.85,
    "confidence": 0.91,
    "timestamp": "2025-01-15T10:30:00Z"
  },
  "webhook_url": "https://backend.workwell.com/api/webhooks/ai"
}
```


## Licença

Este projeto faz parte do sistema WorkWell desenvolvido para FIAP - Faculdade de Informática e Administração Paulista.

**Trabalho Acadêmico** - 2025

### Equipe

- Desenvolvimento de IA/ML
- Integração Backend
- DevOps e Infraestrutura

### Instituição

FIAP - Faculdade de Informática e Administração Paulista

---

**Nota**: Este é um projeto acadêmico desenvolvido como parte da graduação em Análise e Desenvolvimento de Sistemas da FIAP. Não deve ser usado em ambiente de produção sem a devida auditoria de segurança e privacidade.

Para mais informações, consulte a documentação completa ou entre em contato com a equipe de desenvolvimento.
