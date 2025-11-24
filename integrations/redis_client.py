"""
Integração com Redis para cache de alta performance.
Gerencia cache de features, predições, embeddings e sessões.
"""

import os
import json
import logging
from typing import Optional, Any, Dict, List
from datetime import datetime, timedelta
from functools import wraps

import redis
from redis.exceptions import RedisError, ConnectionError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedisManager:
    """Gerenciador de conexões e operações com Redis."""

    def __init__(self, redis_url: Optional[str] = None):
        """
        Inicializa gerenciador Redis.

        Args:
            redis_url: URL de conexão Redis (ex: redis://localhost:6379/0)
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")

        try:
            self.client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                max_connections=50
            )
            self.client.ping()
            logger.info("Redis conectado: %s", self.redis_url)
        except (RedisError, ConnectionError) as e:
            logger.error(f"Erro ao conectar Redis: {e}")
            logger.warning("Redis não disponível - sistema funcionará sem cache")
            self.client = None

    def is_available(self) -> bool:
        """Verifica se Redis está disponível."""
        if self.client is None:
            return False
        try:
            return self.client.ping()
        except:
            return False

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Armazena valor no Redis.

        Args:
            key: Chave
            value: Valor (será serializado para JSON se não for string)
            ttl: Time to live em segundos (None = sem expiração)

        Returns:
            True se sucesso
        """
        if not self.is_available():
            return False

        try:
            if not isinstance(value, str):
                value = json.dumps(value)

            if ttl:
                return self.client.setex(key, ttl, value)
            else:
                return self.client.set(key, value)
        except Exception as e:
            logger.error(f"Erro ao setar {key}: {e}")
            return False

    def get(self, key: str, deserialize: bool = True) -> Optional[Any]:
        """
        Busca valor do Redis.

        Args:
            key: Chave
            deserialize: Se deve deserializar JSON

        Returns:
            Valor ou None se não encontrado
        """
        if not self.is_available():
            return None

        try:
            value = self.client.get(key)
            if value is None:
                return None

            if deserialize:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value
        except Exception as e:
            logger.error(f"Erro ao buscar {key}: {e}")
            return None

    def delete(self, *keys: str) -> int:
        """Deleta uma ou mais chaves."""
        if not self.is_available():
            return 0

        try:
            return self.client.delete(*keys)
        except Exception as e:
            logger.error(f"Erro ao deletar chaves: {e}")
            return 0

    def exists(self, key: str) -> bool:
        """Verifica se chave existe."""
        if not self.is_available():
            return False

        try:
            return self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Erro ao verificar {key}: {e}")
            return False

    def expire(self, key: str, seconds: int) -> bool:
        """Define TTL para chave existente."""
        if not self.is_available():
            return False

        try:
            return self.client.expire(key, seconds)
        except Exception as e:
            logger.error(f"Erro ao setar TTL em {key}: {e}")
            return False

    def hset(self, name: str, key: str, value: Any) -> int:
        """Seta campo em hash."""
        if not self.is_available():
            return 0

        try:
            if not isinstance(value, str):
                value = json.dumps(value)
            return self.client.hset(name, key, value)
        except Exception as e:
            logger.error(f"Erro ao hset {name}.{key}: {e}")
            return 0

    def hget(self, name: str, key: str, deserialize: bool = True) -> Optional[Any]:
        """Busca campo de hash."""
        if not self.is_available():
            return None

        try:
            value = self.client.hget(name, key)
            if value is None:
                return None

            if deserialize:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value
        except Exception as e:
            logger.error(f"Erro ao hget {name}.{key}: {e}")
            return None

    def hgetall(self, name: str, deserialize: bool = True) -> Dict:
        """Busca todos os campos de um hash."""
        if not self.is_available():
            return {}

        try:
            data = self.client.hgetall(name)
            if deserialize:
                return {k: json.loads(v) if v else None for k, v in data.items()}
            return data
        except Exception as e:
            logger.error(f"Erro ao hgetall {name}: {e}")
            return {}

    def hincrby(self, name: str, key: str, amount: int = 1) -> int:
        """Incrementa campo numérico em hash."""
        if not self.is_available():
            return 0

        try:
            return self.client.hincrby(name, key, amount)
        except Exception as e:
            logger.error(f"Erro ao hincrby {name}.{key}: {e}")
            return 0

    def hincrbyfloat(self, name: str, key: str, amount: float) -> float:
        """Incrementa campo float em hash."""
        if not self.is_available():
            return 0.0

        try:
            return self.client.hincrbyfloat(name, key, amount)
        except Exception as e:
            logger.error(f"Erro ao hincrbyfloat {name}.{key}: {e}")
            return 0.0

    def lpush(self, name: str, *values: Any) -> int:
        """Adiciona valores no início da lista."""
        if not self.is_available():
            return 0

        try:
            serialized = [json.dumps(v) if not isinstance(v, str) else v for v in values]
            return self.client.lpush(name, *serialized)
        except Exception as e:
            logger.error(f"Erro ao lpush {name}: {e}")
            return 0

    def rpush(self, name: str, *values: Any) -> int:
        """Adiciona valores no final da lista."""
        if not self.is_available():
            return 0

        try:
            serialized = [json.dumps(v) if not isinstance(v, str) else v for v in values]
            return self.client.rpush(name, *serialized)
        except Exception as e:
            logger.error(f"Erro ao rpush {name}: {e}")
            return 0

    def lrange(self, name: str, start: int, end: int, deserialize: bool = True) -> List[Any]:
        """Busca range de lista."""
        if not self.is_available():
            return []

        try:
            values = self.client.lrange(name, start, end)
            if deserialize:
                return [json.loads(v) if v else None for v in values]
            return values
        except Exception as e:
            logger.error(f"Erro ao lrange {name}: {e}")
            return []

    @staticmethod
    def key_user_features(user_id: int, date: str) -> str:
        """Chave para features calculadas do usuário."""
        return f"user:{user_id}:features:{date}"

    @staticmethod
    def key_prediction_latest(user_id: int) -> str:
        """Chave para última predição do usuário."""
        return f"prediction:user:{user_id}:latest"

    @staticmethod
    def key_embedding_text(text_hash: str) -> str:
        """Chave para embedding de texto."""
        return f"embedding:text:{text_hash}"

    @staticmethod
    def key_chat_session(session_id: str) -> str:
        """Chave para sessão de chat."""
        return f"chat:session:{session_id}"

    @staticmethod
    def key_bandit_arm(item_id: str) -> str:
        """Chave para estatísticas de Multi-Armed Bandit."""
        return f"bandit:arm:{item_id}"

    @staticmethod
    def key_ratelimit(user_id: int) -> str:
        """Chave para rate limiting."""
        return f"ratelimit:user:{user_id}"


def cached(ttl: int = 3600, key_prefix: str = "cache"):
    """
    Decorator para cachear resultados de funções.

    Args:
        ttl: Time to live em segundos
        key_prefix: Prefixo da chave de cache

    Usage:
        @cached(ttl=1800, key_prefix="predictions")
        def get_prediction(user_id):
            return result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            redis_manager = get_redis_manager()
            if not redis_manager.is_available():
                return func(*args, **kwargs)

            cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"

            cached_value = redis_manager.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache HIT: {cache_key}")
                return cached_value

            logger.debug(f"Cache MISS: {cache_key}")
            result = func(*args, **kwargs)

            redis_manager.set(cache_key, result, ttl=ttl)

            return result
        return wrapper
    return decorator


class FeatureCache:
    """Cache especializado para features calculadas."""

    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager

    def get_features(self, user_id: int, date: str) -> Optional[Dict]:
        """Busca features do usuário em uma data."""
        key = RedisManager.key_user_features(user_id, date)
        return self.redis.get(key)

    def set_features(self, user_id: int, date: str, features: Dict, ttl: int = 86400) -> bool:
        """Armazena features (TTL padrão: 24 horas)."""
        key = RedisManager.key_user_features(user_id, date)
        return self.redis.set(key, features, ttl=ttl)


class PredictionCache:
    """Cache especializado para predições."""

    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager

    def get_latest(self, user_id: int) -> Optional[Dict]:
        """Busca última predição do usuário."""
        key = RedisManager.key_prediction_latest(user_id)
        return self.redis.get(key)

    def set_latest(self, user_id: int, prediction: Dict, ttl: int = 3600) -> bool:
        """Armazena última predição (TTL padrão: 1 hora)."""
        key = RedisManager.key_prediction_latest(user_id)
        return self.redis.set(key, prediction, ttl=ttl)


class BanditCache:
    """Cache especializado para Multi-Armed Bandit."""

    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager

    def get_arm_stats(self, item_id: str) -> Dict:
        """Busca estatísticas de um braço (item)."""
        key = RedisManager.key_bandit_arm(item_id)
        stats = self.redis.hgetall(key)
        if not stats:
            return {"count": 0, "reward_sum": 0.0, "avg_reward": 0.5}
        return {
            "count": int(stats.get("count", 0)),
            "reward_sum": float(stats.get("reward_sum", 0.0)),
            "avg_reward": float(stats.get("avg_reward", 0.5))
        }

    def update_arm(self, item_id: str, reward: float) -> None:
        """Atualiza estatísticas de um braço com nova recompensa."""
        key = RedisManager.key_bandit_arm(item_id)

        self.redis.hincrby(key, "count", 1)

        self.redis.hincrbyfloat(key, "reward_sum", reward)

        stats = self.get_arm_stats(item_id)
        new_avg = stats["reward_sum"] / stats["count"] if stats["count"] > 0 else 0.5
        self.redis.hset(key, "avg_reward", new_avg)


class RateLimiter:
    """Rate limiter usando Redis."""

    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager

    def is_allowed(self, user_id: int, max_requests: int = 100, window: int = 60) -> bool:
        """
        Verifica se usuário pode fazer request (sliding window).

        Args:
            user_id: ID do usuário
            max_requests: Máximo de requests no window
            window: Janela de tempo em segundos

        Returns:
            True se permitido
        """
        key = RedisManager.key_ratelimit(user_id)

        try:
            count = self.redis.client.incr(key)
            if count == 1:
                self.redis.expire(key, window)

            return count <= max_requests
        except:
            return True


_redis_manager_instance: Optional[RedisManager] = None

def get_redis_manager() -> RedisManager:
    """Retorna instância global do RedisManager (singleton)."""
    global _redis_manager_instance
    if _redis_manager_instance is None:
        _redis_manager_instance = RedisManager()
    return _redis_manager_instance


if __name__ == "__main__":
    redis_mgr = get_redis_manager()

    if redis_mgr.is_available():
        print("✓ Redis conectado!")

        redis_mgr.set("test:key", {"data": "teste"}, ttl=10)
        result = redis_mgr.get("test:key")
        print(f"Teste get/set: {result}")

        feature_cache = FeatureCache(redis_mgr)
        feature_cache.set_features(123, "2025-01-15", {
            "stress_ma_7d": 0.75,
            "wellbeing_composite": 0.6
        })
        features = feature_cache.get_features(123, "2025-01-15")
        print(f"Features em cache: {features}")

        bandit_cache = BanditCache(redis_mgr)
        bandit_cache.update_arm("mind_001", 0.8)
        bandit_cache.update_arm("mind_001", 0.9)
        stats = bandit_cache.get_arm_stats("mind_001")
        print(f"Bandit stats: {stats}")

        print("\n✓ Todos os testes passaram!")
    else:
        print("✗ Redis não disponível")
