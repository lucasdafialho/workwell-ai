# WorkWell AI - Guia Rápido de Início

## 🚀 Início Rápido

### 1. Instalação
```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar setup
python setup.py
```

### 2. Configuração
Copie `.env.example` para `.env` e configure:
- `GEMINI_API_KEY` ou `OPENAI_API_KEY`
- `DATABASE_URL` (opcional para demonstração)
- `REDIS_URL` (opcional)

### 3. Executar Pipeline ETL
```bash
python main.py etl
```

### 4. Treinar Modelos
```bash
# Treinar modelo de burnout
python main.py train --model burnout
```

### 5. Iniciar API
```bash
python main.py api --port 8000
```

Acesse: `http://localhost:8000/docs`

### 6. Testar API
```bash
python main.py test
```

## 📚 Estrutura do Projeto

```
workwell-ai/
├── api/              # API FastAPI
├── models/           # Modelos de ML/DL
├── pipelines/        # Pipelines ETL
├── services/         # Serviços de IA
├── vision/           # Visão computacional
├── mlops/            # Pipeline MLOps
├── explainability/   # SHAP/LIME
├── privacy/          # Privacidade e segurança
├── monitoring/       # Monitoramento
├── integrations/     # Integrações externas
├── notebooks/        # Notebooks Jupyter
└── examples/         # Exemplos de uso
```

## 🎯 Principais Funcionalidades

1. **Predição de Burnout**: Modelo LSTM para prever risco
2. **Análise de Sentimento**: BERT para análise de textos
3. **IA Generativa**: Chatbot de suporte emocional
4. **Recomendações**: Sistema híbrido de recomendações
5. **Previsão Temporal**: Prophet para séries temporais
6. **Visão Computacional**: Detecção de fadiga
7. **Explicabilidade**: SHAP/LIME para interpretação
8. **Privacidade**: Differential privacy e federated learning

## 📖 Documentação

- `README.md`: Visão geral completa
- `ARCHITECTURE.md`: Arquitetura detalhada
- `api/main.py`: Documentação Swagger automática
- `notebooks/`: Notebooks demonstrativos

## 🔧 Comandos Úteis

```bash
# Setup completo
python setup.py

# Pipeline ETL
python main.py etl

# Treinar todos os modelos
python main.py train --model all

# Iniciar API
python main.py api

# Testar API
python main.py test

# Usar API diretamente
python examples/api_usage.py
```

## ⚠️ Notas Importantes

1. **Modelos NLP**: Execute `python -m spacy download pt_core_news_sm` para modelos em português
2. **GPU**: Modelos podem usar GPU se disponível (CUDA)
3. **API Keys**: Configure no arquivo `.env`
4. **Dados**: Coloque dados em `data/raw/` para processamento

## 🐛 Troubleshooting

- **Erro de importação**: Verifique se todas as dependências estão instaladas
- **API não inicia**: Verifique se a porta está disponível
- **Modelos não carregam**: Execute o treinamento primeiro
- **Erro de memória**: Reduza batch_size nos modelos

## 📞 Suporte

Para dúvidas ou problemas, consulte:
- Documentação em `ARCHITECTURE.md`
- Notebooks em `notebooks/`
- Exemplos em `examples/`

