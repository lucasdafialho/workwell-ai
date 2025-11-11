# Guia Rápido: Como Treinar os Modelos

## 🎯 Método Mais Simples (Recomendado)

```bash
# 1. Gerar dados sintéticos
python pipelines/generate_data.py

# 2. Treinar todos os modelos
python pipelines/train_all.py --skip-data
```

Ou tudo de uma vez:
```bash
python pipelines/train_all.py
```

## 📊 O Que Cada Script Faz

### `generate_data.py`
- Gera dados sintéticos realistas de check-ins
- Cria dados de interações com recomendações
- Salva em `data/raw/`

### `train_burnout.py`
- Processa dados com pipeline ETL
- Cria sequências temporais
- Treina modelo LSTM
- Salva modelo em `models/storage/`

### `train_sentiment.py`
- Carrega modelo BERT pré-treinado
- Valida funcionamento
- Pronto para uso imediato

### `train_recommendation.py`
- Treina sistema de recomendação
- Prepara modelos collaborative e content-based

## 🔄 Fluxo Completo

```bash
# Passo 1: Gerar dados (se necessário)
python pipelines/generate_data.py

# Passo 2: Treinar modelos
python pipelines/train_all.py --skip-data

# Passo 3: Verificar modelos treinados
ls -lh models/storage/

# Passo 4: Iniciar API
python main.py api

# Passo 5: Testar
python main.py test
```

## ⚡ Comandos Rápidos

```bash
# Tudo de uma vez
python pipelines/train_all.py

# Apenas modelo de burnout
python pipelines/train_burnout.py

# Apenas gerar dados
python pipelines/generate_data.py
```

## 📝 Dados Necessários

### Para Burnout (LSTM):
- Arquivo: `data/raw/checkins.csv`
- Colunas: `usuario_id`, `data_checkin`, `nivel_stress`, `horas_trabalhadas`, `horas_sono`, `score_bemestar`
- Mínimo: 30 dias por usuário, 10+ usuários

### Para Recomendação:
- Arquivo: `data/raw/interactions.csv`
- Colunas: `user_id`, `item_id`, `rating`, `timestamp`
- Mínimo: 100+ interações

## ✅ Verificação

Após treinar, verifique:
```bash
# Verificar se modelo foi salvo
ls models/storage/best_burnout_model.pt

# Testar carregamento
python -c "from models.burnout.lstm_model import BurnoutPredictor; p = BurnoutPredictor(); p.load_model('models/storage/best_burnout_model.pt'); print('OK!')"
```

## 🆘 Problemas Comuns

**"Arquivo não encontrado"**
→ Execute `python pipelines/generate_data.py` primeiro

**"Out of memory"**
→ Reduza batch_size em `train_burnout.py`

**"Dados insuficientes"**
→ Gere mais dados ou reduza sequence_length

## 📚 Mais Informações

Veja `TRAINING_GUIDE.md` para guia completo e detalhado.

