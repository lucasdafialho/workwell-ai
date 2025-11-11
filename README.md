# WorkWell AI - Módulo de Inteligência Artificial

Sistema inteligente de prevenção de burnout e otimização de bem-estar corporativo utilizando Deep Learning, Visão Computacional e IA Generativa.

## 🏗️ Arquitetura

O módulo de IA está estruturado em três camadas principais:

1. **Camada de Coleta e Preparação de Dados**: Processa informações dos check-ins diários, métricas de saúde e padrões de trabalho
2. **Camada de Modelos de Machine Learning**: Implementa modelos para análise preditiva e classificação
3. **Camada de Serviços de IA**: Expõe serviços através de APIs RESTful integradas com o backend principal

## 📁 Estrutura do Projeto

```
workwell-ai/
├── api/                    # API FastAPI
├── models/                 # Modelos de ML/DL
│   ├── burnout/           # Modelo LSTM para burnout
│   ├── sentiment/         # Modelo BERT para sentimento
│   ├── fatigue/           # Modelo CNN para fadiga
│   └── timeseries/        # Modelo Prophet para séries temporais
├── pipelines/              # Pipelines de ETL e treinamento
├── services/               # Serviços de IA
│   ├── generative/        # IA generativa (Gemini/GPT)
│   ├── recommendation/    # Sistema de recomendação
│   └── nlp/               # Processamento de linguagem natural
├── mlops/                  # Pipeline MLOps (MLflow)
├── vision/                 # Visão computacional
├── explainability/         # SHAP/LIME para explicabilidade
├── privacy/                # Privacidade e segurança
├── monitoring/             # Monitoramento e métricas
├── integrations/           # Integrações externas
├── notebooks/              # Notebooks Jupyter demonstrativos
└── utils/                  # Utilitários e helpers
```

## 🚀 Instalação

```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Configurar variáveis de ambiente
cp .env.example .env
# Editar .env com suas credenciais
```

## 🔧 Configuração

Configure as variáveis de ambiente no arquivo `.env`:

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key

# Database
DATABASE_URL=postgresql://user:pass@localhost/workwell
REDIS_URL=redis://localhost:6379

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Model Storage
MODEL_STORAGE_PATH=./models/storage
```

## 📊 Componentes Principais

### 1. Modelo de Predição de Burnout (LSTM)
Rede neural profunda para prever risco de burnout usando padrões temporais.

### 2. Visão Computacional para Fadiga
Detecção de sinais de fadiga em videochamadas usando MediaPipe e CNN.

### 3. IA Generativa para Suporte Emocional
Chatbot terapêutico com Gemini/GPT usando RAG e LangChain.

### 4. Análise de Sentimento Avançada
Modelo BERT fine-tunado em português para análise profunda de sentimentos.

### 5. Sistema de Recomendação
Engine híbrida combinando collaborative filtering e content-based filtering.

### 6. Séries Temporais
Modelo Prophet para previsão de tendências de bem-estar.

## 🎯 Uso

### Treinar Modelos (Primeiro Passo)

```bash
# Método mais simples: gerar dados e treinar tudo
python pipelines/train_all.py

# Ou passo a passo:
# 1. Gerar dados sintéticos
python pipelines/generate_data.py

# 2. Treinar todos os modelos
python pipelines/train_all.py --skip-data

# 3. Treinar modelo específico
python pipelines/train_burnout.py
python pipelines/train_sentiment.py
python pipelines/train_recommendation.py
```

**📖 Veja `HOW_TO_TRAIN.md` ou `TRAINING_GUIDE.md` para guias detalhados.**

### Iniciar API FastAPI

```bash
# Após treinar os modelos
python main.py api
# ou
uvicorn api.main:app --reload --port 8000
```

### Executar Notebooks

```bash
jupyter notebook notebooks/
```

## 📈 Monitoramento

Acesse o dashboard de monitoramento em: `http://localhost:8000/docs`

## 🔒 Privacidade e Segurança

- Differential Privacy para proteção de dados individuais
- Federated Learning para treinamento distribuído
- Criptografia de modelos e dados
- Anonimização de dados antes do treinamento
- Conformidade com LGPD

## 📚 Documentação

- `HOW_TO_TRAIN.md`: Guia rápido de treinamento
- `TRAINING_GUIDE.md`: Guia completo de treinamento
- `ARCHITECTURE.md`: Arquitetura detalhada
- `QUICKSTART.md`: Início rápido
- `notebooks/`: Notebooks Jupyter demonstrativos

## 🤝 Integração

O módulo de IA se integra com o backend .NET através de APIs RESTful. Veja `api/integration.py` para detalhes de integração.

## 📝 Licença

Este projeto faz parte do sistema WorkWell desenvolvido para FIAP.

