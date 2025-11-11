# Guia de Treinamento de Modelos - WorkWell AI

## 📊 Dados para Treinamento

### Opção 1: Dados Sintéticos (Recomendado para demonstração)

O projeto inclui um gerador de dados sintéticos que cria dados realistas para treinamento:

```bash
# Gerar dados sintéticos
python pipelines/generate_data.py
```

Isso cria:
- `data/raw/checkins.csv`: ~9.000 check-ins de 50 usuários ao longo de 180 dias
- `data/raw/interactions.csv`: Dados de interações com recomendações

**Características dos dados sintéticos:**
- Padrões realistas de stress, sono e trabalho
- Sazonalidade semanal e mensal
- Tendências temporais (melhorando/piorando/estável)
- Valores faltantes simulados
- Correlações entre variáveis

### Opção 2: Dados Reais do Banco de Dados

Para usar dados reais do WorkWell:

1. **Exportar dados do banco PostgreSQL:**
```sql
-- Exportar check-ins
COPY (
    SELECT 
        id, usuario_id, data_checkin, nivel_stress, 
        horas_trabalhadas, horas_sono, sentimento, 
        observacoes, score_bemestar
    FROM checkins_diarios
    ORDER BY usuario_id, data_checkin
) TO '/caminho/para/data/raw/checkins.csv' WITH CSV HEADER;
```

2. **Salvar em `data/raw/checkins.csv`**

3. **Executar pipeline ETL:**
```bash
python pipelines/etl_pipeline.py
```

## 🚀 Treinamento dos Modelos

### Treinar Todos os Modelos (Recomendado)

```bash
# Treinar tudo de uma vez (gera dados se necessário)
python pipelines/train_all.py

# Ou pular geração de dados se já existirem
python pipelines/train_all.py --skip-data
```

### Treinar Modelos Individuais

#### 1. Modelo de Predição de Burnout (LSTM)

```bash
# Com dados padrão
python pipelines/train_burnout.py

# Com dados customizados
python pipelines/train_burnout.py \
    --data data/raw/meus_checkins.csv \
    --output models/storage/meu_modelo.pt \
    --epochs 100 \
    --batch-size 64
```

**Requisitos:**
- Dados de check-ins com pelo menos 30 dias por usuário
- Mínimo de 10-20 usuários recomendado
- Colunas necessárias: `usuario_id`, `data_checkin`, `nivel_stress`, `horas_trabalhadas`, `horas_sono`, `score_bemestar`

**Tempo estimado:** 10-30 minutos (depende do hardware)

#### 2. Modelo de Análise de Sentimento (BERT)

```bash
python pipelines/train_sentiment.py
```

**Nota:** O modelo BERT já vem pré-treinado. Este script apenas valida e prepara o modelo.

**Para fine-tuning com dados específicos:**
- Use o notebook `notebooks/sentiment_finetuning.ipynb` (criar se necessário)
- Requer dataset de textos rotulados em português

#### 3. Sistema de Recomendação

```bash
python pipelines/train_recommendation.py
```

**Requisitos:**
- Dados de interações: `user_id`, `item_id`, `rating`, `timestamp`
- Mínimo de 100-200 interações recomendado

**Nota:** O sistema melhora continuamente com feedback dos usuários.

## 📋 Checklist de Treinamento

- [ ] Dados disponíveis em `data/raw/`
- [ ] Ambiente virtual ativado
- [ ] Dependências instaladas (`pip install -r requirements.txt`)
- [ ] GPU disponível (opcional, mas recomendado para LSTM)
- [ ] Espaço em disco suficiente (~500MB para modelos)

## 🔧 Configurações Avançadas

### Ajustar Hiperparâmetros do LSTM

Edite `models/burnout/lstm_model.py`:

```python
predictor = BurnoutPredictor(config={
    'hidden_size': 256,      # Tamanho da camada oculta
    'num_layers': 3,         # Número de camadas LSTM
    'dropout': 0.4           # Taxa de dropout
})
```

### Usar GPU para Treinamento

O código detecta automaticamente GPU se disponível. Para forçar CPU:

```python
# Em lstm_model.py
self.device = torch.device('cpu')
```

### Treinar com Menos Dados

Para datasets pequenos, ajuste:
- `sequence_length`: Reduzir de 30 para 15-20
- `batch_size`: Reduzir para 16 ou 8
- `epochs`: Aumentar para compensar

## 📊 Monitoramento do Treinamento

### Durante o Treinamento

O script mostra:
- Loss e accuracy por época
- Early stopping automático
- Melhor modelo salvo automaticamente

### Após o Treinamento

Verifique:
- `models/storage/best_burnout_model.pt`: Modelo treinado
- Logs em console: Métricas finais
- MLflow (se configurado): Experimentos registrados

## 🐛 Troubleshooting

### Erro: "Dados insuficientes"
- **Solução:** Gere mais dados ou reduza `sequence_length`

### Erro: "Out of memory"
- **Solução:** Reduza `batch_size` ou `sequence_length`

### Erro: "Modelo não converge"
- **Solução:** Ajuste learning rate ou adicione mais dados

### Erro: "CUDA out of memory"
- **Solução:** Use CPU ou reduza batch_size

## 📈 Próximos Passos Após Treinamento

1. **Validar modelo:**
```bash
python -c "from models.burnout.lstm_model import BurnoutPredictor; p = BurnoutPredictor(); p.load_model('models/storage/best_burnout_model.pt'); print('Modelo carregado!')"
```

2. **Iniciar API:**
```bash
python main.py api
```

3. **Testar predições:**
```bash
python main.py test
```

4. **Usar em produção:**
   - Integrar com backend .NET
   - Configurar monitoramento
   - Implementar retreinamento automático

## 💡 Dicas

- **Dados sintéticos são suficientes para demonstração**
- **Para produção, use dados reais do banco**
- **Treine periodicamente com novos dados**
- **Monitore performance em produção**
- **Use MLflow para versionamento**

## 📚 Recursos Adicionais

- `ARCHITECTURE.md`: Arquitetura detalhada
- `README.md`: Visão geral do projeto
- `notebooks/`: Notebooks demonstrativos
- Código comentado em todos os módulos

