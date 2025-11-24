"""
Script de inicialização do banco de dados PostgreSQL.
Cria todas as tabelas e pode popular com dados de exemplo.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import argparse
import logging
from integrations.database import get_database_manager, CheckIn, UserProfile
from integrations.redis_client import get_redis_manager
from datetime import datetime, timedelta
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_postgresql(reset: bool = False, seed: bool = False):
    """
    Inicializa PostgreSQL.

    Args:
        reset: Se deve resetar (dropar e recriar) as tabelas
        seed: Se deve popular com dados de exemplo
    """
    logger.info("=" * 60)
    logger.info("INICIALIZANDO POSTGRESQL")
    logger.info("=" * 60)

    try:
        db_manager = get_database_manager()

        if reset:
            logger.warning("RESETANDO banco de dados (deletando todas as tabelas)...")
            response = input("Tem certeza? Digite 'SIM' para confirmar: ")
            if response != "SIM":
                logger.info("Operação cancelada")
                return
            db_manager.drop_tables()

        logger.info("Criando tabelas...")
        db_manager.create_tables()
        logger.info("✓ Tabelas criadas com sucesso!")

        if seed:
            logger.info("Populando banco com dados de exemplo...")
            seed_database(db_manager)

        logger.info("=" * 60)
        logger.info("POSTGRESQL PRONTO!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"✗ Erro ao inicializar PostgreSQL: {e}")
        logger.error("Verifique se o PostgreSQL está rodando e as credenciais estão corretas")
        sys.exit(1)


def init_redis():
    """Testa conexão com Redis."""
    logger.info("=" * 60)
    logger.info("TESTANDO REDIS")
    logger.info("=" * 60)

    try:
        redis_manager = get_redis_manager()

        if redis_manager.is_available():
            logger.info("✓ Redis conectado!")

            redis_manager.set("test:init", {"timestamp": datetime.now().isoformat()}, ttl=60)
            result = redis_manager.get("test:init")

            if result:
                logger.info(f"✓ Teste de escrita/leitura: OK")
                logger.info(f"  Resultado: {result}")
            else:
                logger.warning("✗ Falha no teste de escrita/leitura")

            redis_manager.delete("test:init")

            logger.info("=" * 60)
            logger.info("REDIS PRONTO!")
            logger.info("=" * 60)
        else:
            raise Exception("Redis não respondeu ao ping")

    except Exception as e:
        logger.error(f"✗ Erro ao conectar Redis: {e}")
        logger.error("Verifique se o Redis está rodando")
        logger.warning("Sistema pode funcionar sem Redis, mas sem cache")


def seed_database(db_manager):
    """Popula banco com dados de exemplo."""
    with db_manager.get_session() as session:
        logger.info("Criando perfis de usuários...")
        for user_id in range(1, 6):
            profile = UserProfile(
                usuario_id=user_id,
                preferred_tags=["stress", "mindfulness", "relaxamento"],
                stress_tolerance=random.randint(5, 8),
                available_time=random.choice([10, 15, 20]),
                notification_preferences={"email": True, "push": True},
                privacy_settings={"share_data": False}
            )
            session.add(profile)

        logger.info("Criando check-ins de exemplo...")
        for user_id in range(1, 6):
            for days_ago in range(30, 0, -1):
                checkin = CheckIn(
                    usuario_id=user_id,
                    data_checkin=datetime.now() - timedelta(days=days_ago),
                    nivel_stress=random.randint(3, 9),
                    horas_trabalhadas=random.uniform(6, 12),
                    horas_sono=random.uniform(5, 8),
                    sentimento=random.choice(["positivo", "neutro", "negativo", "cansado"]),
                    observacoes=f"Check-in de exemplo do dia {days_ago}",
                    score_bemestar=random.uniform(40, 85)
                )
                session.add(checkin)

        session.commit()
        logger.info("✓ Dados de exemplo criados!")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Inicialização do banco de dados WorkWell AI"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reseta o banco (DELETA todas as tabelas e dados)"
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Popula banco com dados de exemplo"
    )
    parser.add_argument(
        "--test-redis",
        action="store_true",
        help="Testa apenas conexão com Redis"
    )
    parser.add_argument(
        "--skip-postgresql",
        action="store_true",
        help="Pula inicialização do PostgreSQL"
    )

    args = parser.parse_args()

    if args.test_redis:
        init_redis()
        return

    if not args.skip_postgresql:
        init_postgresql(reset=args.reset, seed=args.seed)

    init_redis()

    print("\n" + "=" * 60)
    print("INICIALIZAÇÃO COMPLETA!")
    print("=" * 60)
    print("\nPróximos passos:")
    print("1. Inicie a API: uvicorn api.main:app --reload")
    print("2. Acesse http://localhost:8000/docs")
    print("3. Teste os endpoints!")


if __name__ == "__main__":
    main()
