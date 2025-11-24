# Setup de Bancos de Dados - WorkWell AI

Guia completo para configurar PostgreSQL e Redis no projeto WorkWell AI.

## Índice

- [PostgreSQL](#postgresql)
- [Redis](#redis)
- [Inicialização Rápida](#inicialização-rápida)
- [Troubleshooting](#troubleshooting)

---

## PostgreSQL

### Instalação

#### Windows

```bash
# Baixe e instale: https://www.postgresql.org/download/windows/
# Ou use Chocolatey:
choco install postgresql

# Inicie o serviço
net start postgresql-x64-14
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib

# Inicie o serviço
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

#### macOS

```bash
# Usando Homebrew
brew install postgresql@14

# Inicie o serviço
brew services start postgresql@14
```

### Configuração

1. **Crie o banco de dados e usuário:**

```bash
# Entre no psql
sudo -u postgres psql

# Ou no Windows (PowerShell como Admin):
psql -U postgres
```

```sql
-- Crie o banco de dados
CREATE DATABASE workwell;

-- Crie o usuário
CREATE USER workwell WITH PASSWORD 'workwell';

-- Conceda privilégios
GRANT ALL PRIVILEGES ON DATABASE workwell TO workwell;

-- Saia
\q
```

2. **Configure a URL de conexão no `.env`:**

```env
DATABASE_URL=postgresql://workwell:workwell@localhost:5432/workwell
```

3. **Teste a conexão:**

```bash
psql -h localhost -U workwell -d workwell
# Senha: workwell
```

---

## Redis

### Instalação

#### Windows

```bash
# Baixe o MSI: https://github.com/microsoftarchive/redis/releases
# Ou use Chocolatey:
choco install redis-64

# Inicie o serviço
redis-server
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install redis-server

# Inicie o serviço
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

#### macOS

```bash
# Usando Homebrew
brew install redis

# Inicie o serviço
brew services start redis
```

### Configuração

1. **Teste a conexão:**

```bash
redis-cli ping
# Deve retornar: PONG
```

2. **Configure a URL no `.env`:**

```env
REDIS_URL=redis://localhost:6379/0
```

### Configurações Opcionais do Redis

Edite `/etc/redis/redis.conf` (Linux) ou `C:\Program Files\Redis\redis.windows.conf` (Windows):

```conf
# Persistência (salva dados em disco)
save 900 1
save 300 10
save 60 10000

# Memória máxima
maxmemory 256mb
maxmemory-policy allkeys-lru

# Bind (escute apenas localhost por segurança)
bind 127.0.0.1
```

Reinicie o Redis após mudanças:

```bash
# Linux
sudo systemctl restart redis-server

# Windows
net stop Redis && net start Redis
```

---

## Inicialização Rápida

### 1. Configure o ambiente

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite as configurações (se necessário)
nano .env  # ou vim, code, notepad++
```

### 2. Instale as dependências Python

```bash
pip install sqlalchemy psycopg2-binary redis python-dotenv
```

### 3. Inicialize o banco de dados

```bash
# Cria as tabelas
python scripts/init_database.py

# Ou com dados de exemplo
python scripts/init_database.py --seed

# Ou resetando tudo (CUIDADO: deleta dados!)
python scripts/init_database.py --reset --seed
```

### 4. Verifique a conexão

```bash
# Teste apenas Redis
python scripts/init_database.py --test-redis

# Ou teste manualmente
python -c "from integrations.database import get_database_manager; db = get_database_manager(); print('PostgreSQL OK!')"
python -c "from integrations.redis_client import get_redis_manager; r = get_redis_manager(); print(f'Redis: {r.is_available()}')"
```

### 5. Inicie a API

```bash
uvicorn api.main:app --reload
```

A API agora irá:
-  Conectar ao PostgreSQL para armazenar dados persistentes
-  Conectar ao Redis para cache de alta performance
-  Criar tabelas automaticamente se não existirem
-  Funcionar mesmo se PostgreSQL/Redis não estiverem disponíveis (modo degradado)

---

## Usando os Bancos

### PostgreSQL - Armazenamento Persistente

**Dados armazenados:**
- Check-ins diários completos
- Predições de burnout com histórico
- Interações com recomendações (para collaborative filtering)
- Análises de sentimento
- Histórico de conversas do chatbot
- Detecções de fadiga
- Perfis de usuários

**Exemplo de query direta:**

```python
from integrations.database import get_database_manager, CheckIn
from datetime import datetime, timedelta

db_manager = get_database_manager()

with db_manager.get_session() as session:
    # Busca check-ins dos últimos 7 dias
    week_ago = datetime.now() - timedelta(days=7)
    checkins = session.query(CheckIn)\
        .filter(CheckIn.data_checkin >= week_ago)\
        .all()

    for checkin in checkins:
        print(f"User {checkin.usuario_id}: Stress {checkin.nivel_stress}/10")
```

**Usando repositórios (recomendado):**

```python
from integrations.database import get_database_manager, CheckInRepository

db_manager = get_database_manager()

with db_manager.get_session() as session:
    # Busca últimos 30 check-ins do usuário
    checkins = CheckInRepository.get_by_user(session, usuario_id=123, limit=30)
```

### Redis - Cache de Alta Performance

**Dados em cache:**
- Features calculadas (TTL: 24h)
- Últimas predições (TTL: 1h)
- Embeddings de texto (TTL: 7 dias)
- Sessões de chat (TTL: 24h)
- Estatísticas do Multi-Armed Bandit (sem TTL)
- Rate limiting (TTL: 1min)

**Exemplo de uso:**

```python
from integrations.redis_client import get_redis_manager, FeatureCache, PredictionCache

redis_mgr = get_redis_manager()

# Cache de features
feature_cache = FeatureCache(redis_mgr)
features = feature_cache.get_features(user_id=123, date="2025-01-15")
if features is None:
    # Calcula features (operação cara)
    features = calculate_expensive_features()
    # Armazena em cache por 24h
    feature_cache.set_features(user_id=123, date="2025-01-15", features=features)

# Cache de predições
prediction_cache = PredictionCache(redis_mgr)
prediction = prediction_cache.get_latest(user_id=123)
if prediction is None:
    # Faz predição
    prediction = model.predict(...)
    # Armazena em cache por 1h
    prediction_cache.set_latest(user_id=123, prediction=prediction)
```

---

## Estrutura das Tabelas

### Principais Tabelas PostgreSQL

```sql
-- Check-ins diários
CREATE TABLE checkins (
    id SERIAL PRIMARY KEY,
    usuario_id INTEGER NOT NULL,
    data_checkin TIMESTAMP NOT NULL,
    nivel_stress INTEGER NOT NULL,
    horas_trabalhadas FLOAT NOT NULL,
    horas_sono FLOAT,
    sentimento VARCHAR(50),
    observacoes TEXT,
    score_bemestar FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Predições de burnout
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    usuario_id INTEGER NOT NULL,
    predicted_class VARCHAR(20) NOT NULL,
    probabilities JSON NOT NULL,
    risk_score FLOAT NOT NULL,
    confidence FLOAT,
    contributing_factors JSON,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Interações (para sistema de recomendação)
CREATE TABLE interactions (
    id SERIAL PRIMARY KEY,
    usuario_id INTEGER NOT NULL,
    item_id VARCHAR(50) NOT NULL,
    item_type VARCHAR(50),
    rating FLOAT,
    completed BOOLEAN DEFAULT FALSE,
    feedback_text TEXT,
    timestamp TIMESTAMP DEFAULT NOW()
);
```

---

## Troubleshooting

### PostgreSQL não conecta

**Erro:** `psycopg2.OperationalError: could not connect to server`

**Soluções:**

1. Verifique se está rodando:
```bash
# Linux
sudo systemctl status postgresql

# Windows
sc query postgresql-x64-14
```

2. Verifique o arquivo pg_hba.conf:
```bash
# Linux: /etc/postgresql/14/main/pg_hba.conf
# Windows: C:\Program Files\PostgreSQL\14\data\pg_hba.conf

# Adicione se necessário:
host    all             all             127.0.0.1/32            md5
```

3. Verifique a porta:
```bash
sudo lsof -i :5432  # Linux
netstat -an | find "5432"  # Windows
```

### Redis não conecta

**Erro:** `redis.exceptions.ConnectionError: Error 10061 connecting to localhost:6379`

**Soluções:**

1. Verifique se está rodando:
```bash
# Tente conectar
redis-cli ping

# Se falhar, inicie o serviço
redis-server  # Modo foreground
# Ou
sudo systemctl start redis-server  # Linux
net start Redis  # Windows
```

2. Verifique a porta:
```bash
sudo lsof -i :6379  # Linux
netstat -an | find "6379"  # Windows
```

### Erro de permissão no PostgreSQL

**Erro:** `permission denied for table X`

**Solução:**

```sql
-- Entre como postgres
sudo -u postgres psql -d workwell

-- Conceda todas as permissões
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO workwell;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO workwell;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO workwell;
```

### Sistema funciona sem bancos?

**SIM!** O WorkWell AI foi projetado para funcionar em modo degradado:

- **Sem PostgreSQL**: Dados não são persistidos, mas predições funcionam
- **Sem Redis**: Sem cache, mas todas as funcionalidades de IA funcionam (mais lento)
- **Sem ambos**: Funciona totalmente em memória (dados perdem ao reiniciar)

Logs de aviso aparecerão, mas a API inicia normalmente.

---

## Docker (Alternativa Fácil)

Se preferir usar Docker para os bancos:

```yaml
# docker-compose.yml (criar na raiz)
version: '3.8'

services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_USER: workwell
      POSTGRES_PASSWORD: workwell
      POSTGRES_DB: workwell
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

```bash
# Inicie os bancos
docker-compose up -d

# Pare os bancos
docker-compose down

# Limpe tudo (DELETA dados!)
docker-compose down -v
```

---

## Monitoramento

### PostgreSQL

```sql
-- Verificar conexões ativas
SELECT count(*) FROM pg_stat_activity;

-- Tamanho do banco
SELECT pg_size_pretty(pg_database_size('workwell'));

-- Tabelas e tamanhos
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
FROM pg_tables WHERE schemaname = 'public';
```

### Redis

```bash
# Informações gerais
redis-cli info

# Uso de memória
redis-cli info memory

# Número de chaves
redis-cli dbsize

# Listar chaves (CUIDADO em produção!)
redis-cli keys "*"

# Monitor em tempo real
redis-cli monitor
```

---

## Backup e Restore

### PostgreSQL

```bash
# Backup
pg_dump -U workwell -h localhost workwell > backup_$(date +%Y%m%d).sql

# Restore
psql -U workwell -h localhost -d workwell < backup_20250124.sql
```

### Redis

```bash
# Backup manual (cria dump.rdb)
redis-cli save

# Ou assíncrono
redis-cli bgsave

# Arquivo gerado em:
# Linux: /var/lib/redis/dump.rdb
# Windows: C:\Program Files\Redis\dump.rdb
```

---

**Pronto! Seus bancos de dados estão configurados.** 

Para mais detalhes, consulte:
- [Documentação PostgreSQL](https://www.postgresql.org/docs/)
- [Documentação Redis](https://redis.io/documentation)
