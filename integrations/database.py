"""
Integração com PostgreSQL usando SQLAlchemy.
Gerencia conexões, sessões e modelos de dados.
"""

import os
import logging
from typing import Generator, Optional
from datetime import datetime
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()


class CheckIn(Base):
    """Modelo para check-ins diários dos usuários."""
    __tablename__ = "checkins"

    id = Column(Integer, primary_key=True, autoincrement=True)
    usuario_id = Column(Integer, nullable=False, index=True)
    data_checkin = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    nivel_stress = Column(Integer, nullable=False)
    horas_trabalhadas = Column(Float, nullable=False)
    horas_sono = Column(Float)
    sentimento = Column(String(50))
    observacoes = Column(Text)
    score_bemestar = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    """Modelo para armazenar predições de burnout."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    usuario_id = Column(Integer, nullable=False, index=True)
    predicted_class = Column(String(20), nullable=False)  # baixo, medio, alto, critico
    probabilities = Column(JSON, nullable=False)  # {"baixo": 0.1, "medio": 0.2, ...}
    risk_score = Column(Float, nullable=False)
    confidence = Column(Float)
    contributing_factors = Column(JSON)  # [{"factor": "stress", "importance": 0.34}, ...]
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class Interaction(Base):
    """Modelo para interações do usuário com recomendações."""
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    usuario_id = Column(Integer, nullable=False, index=True)
    item_id = Column(String(50), nullable=False, index=True)
    item_type = Column(String(50))  # exercise, meditation, break_time, content
    rating = Column(Float)  # 0-5
    completed = Column(Boolean, default=False)
    feedback_text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class SentimentAnalysis(Base):
    """Modelo para análises de sentimento."""
    __tablename__ = "sentiment_analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    usuario_id = Column(Integer, nullable=False, index=True)
    text = Column(Text, nullable=False)
    sentiment = Column(String(20))  # positivo, neutro, negativo
    score = Column(Float)
    emotions = Column(JSON)  # {"alegria": 0.8, "tristeza": 0.2, ...}
    risk_level = Column(String(20))  # baixo, medio, alto
    risk_keywords = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class ChatHistory(Base):
    """Modelo para histórico de conversas do chatbot."""
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    usuario_id = Column(Integer, nullable=False, index=True)
    session_id = Column(String(100), index=True)
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    sources = Column(JSON)  # ["burnout", "mindfulness", ...]
    confidence = Column(Float)
    sentiment_detected = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class FatigueDetection(Base):
    """Modelo para detecções de fadiga facial."""
    __tablename__ = "fatigue_detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    usuario_id = Column(Integer, nullable=False, index=True)
    fatigue_score = Column(Float, nullable=False)
    fatigue_level = Column(String(20))  # baixo, medio, alto, critico
    blink_rate = Column(Float)
    ear_score = Column(Float)  # Eye Aspect Ratio
    yawn_detected = Column(Boolean, default=False)
    head_tilt = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class UserProfile(Base):
    """Modelo para perfis de usuários."""
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    usuario_id = Column(Integer, unique=True, nullable=False, index=True)
    preferred_tags = Column(JSON)  # ["stress", "relaxamento", ...]
    stress_tolerance = Column(Integer)  # 1-10
    available_time = Column(Integer)  # minutos por dia
    notification_preferences = Column(JSON)
    privacy_settings = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DatabaseManager:
    """Gerenciador de conexões e operações com PostgreSQL."""

    def __init__(self, database_url: Optional[str] = None):
        """
        Inicializa gerenciador de banco de dados.

        Args:
            database_url: URL de conexão PostgreSQL (ex: postgresql://user:pass@localhost/db)
        """
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://workwell:workwell@localhost:5432/workwell"
        )

        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # Verifica conexão antes de usar
            echo=False
        )

        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        logger.info("DatabaseManager inicializado: %s", self.database_url.split('@')[-1])

    def create_tables(self):
        """Cria todas as tabelas no banco de dados."""
        logger.info("Criando tabelas no banco de dados...")
        Base.metadata.create_all(bind=self.engine)
        logger.info("Tabelas criadas com sucesso!")

    def drop_tables(self):
        """Remove todas as tabelas (CUIDADO: deleta dados!)."""
        logger.warning("Removendo todas as tabelas do banco de dados...")
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Tabelas removidas!")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager para sessões de banco de dados.

        Usage:
            with db_manager.get_session() as session:
                session.add(checkin)
                session.commit()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Erro na sessão do banco: {e}")
            raise
        finally:
            session.close()

    def get_db(self) -> Generator[Session, None, None]:
        """
        Dependency para FastAPI.

        Usage:
            @app.get("/users")
            def get_users(db: Session = Depends(db_manager.get_db)):
                return db.query(CheckIn).all()
        """
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()


class CheckInRepository:
    """Repositório para operações com check-ins."""

    @staticmethod
    def create(session: Session, checkin_data: dict) -> CheckIn:
        """Cria novo check-in."""
        checkin = CheckIn(**checkin_data)
        session.add(checkin)
        session.flush()
        return checkin

    @staticmethod
    def get_by_user(session: Session, usuario_id: int, limit: int = 30) -> list[CheckIn]:
        """Busca últimos N check-ins do usuário."""
        return session.query(CheckIn)\
            .filter(CheckIn.usuario_id == usuario_id)\
            .order_by(CheckIn.data_checkin.desc())\
            .limit(limit)\
            .all()

    @staticmethod
    def get_by_date_range(session: Session, usuario_id: int, start_date: datetime, end_date: datetime) -> list[CheckIn]:
        """Busca check-ins em range de datas."""
        return session.query(CheckIn)\
            .filter(
                CheckIn.usuario_id == usuario_id,
                CheckIn.data_checkin >= start_date,
                CheckIn.data_checkin <= end_date
            )\
            .order_by(CheckIn.data_checkin.asc())\
            .all()


class PredictionRepository:
    """Repositório para operações com predições."""

    @staticmethod
    def create(session: Session, prediction_data: dict) -> Prediction:
        """Cria nova predição."""
        prediction = Prediction(**prediction_data)
        session.add(prediction)
        session.flush()
        return prediction

    @staticmethod
    def get_latest(session: Session, usuario_id: int) -> Optional[Prediction]:
        """Busca última predição do usuário."""
        return session.query(Prediction)\
            .filter(Prediction.usuario_id == usuario_id)\
            .order_by(Prediction.created_at.desc())\
            .first()

    @staticmethod
    def get_high_risk_users(session: Session, threshold: float = 0.7) -> list[Prediction]:
        """Busca usuários com alto risco."""
        return session.query(Prediction)\
            .filter(Prediction.risk_score >= threshold)\
            .order_by(Prediction.created_at.desc())\
            .all()


class InteractionRepository:
    """Repositório para operações com interações."""

    @staticmethod
    def create(session: Session, interaction_data: dict) -> Interaction:
        """Cria nova interação."""
        interaction = Interaction(**interaction_data)
        session.add(interaction)
        session.flush()
        return interaction

    @staticmethod
    def get_by_user(session: Session, usuario_id: int) -> list[Interaction]:
        """Busca todas as interações do usuário."""
        return session.query(Interaction)\
            .filter(Interaction.usuario_id == usuario_id)\
            .order_by(Interaction.timestamp.desc())\
            .all()

    @staticmethod
    def get_by_item(session: Session, item_id: str) -> list[Interaction]:
        """Busca todas as interações com um item."""
        return session.query(Interaction)\
            .filter(Interaction.item_id == item_id)\
            .all()


_db_manager_instance: Optional[DatabaseManager] = None

def get_database_manager() -> DatabaseManager:
    """Retorna instância global do DatabaseManager (singleton)."""
    global _db_manager_instance
    if _db_manager_instance is None:
        _db_manager_instance = DatabaseManager()
    return _db_manager_instance


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gerenciamento do banco de dados")
    parser.add_argument("--create", action="store_true", help="Cria tabelas")
    parser.add_argument("--drop", action="store_true", help="Remove tabelas")
    parser.add_argument("--reset", action="store_true", help="Remove e recria tabelas")

    args = parser.parse_args()

    db_manager = get_database_manager()

    if args.reset:
        db_manager.drop_tables()
        db_manager.create_tables()
        print("✓ Banco de dados resetado!")
    elif args.drop:
        db_manager.drop_tables()
        print("✓ Tabelas removidas!")
    elif args.create:
        db_manager.create_tables()
        print("✓ Tabelas criadas!")
    else:
        print("Uso: python database.py --create|--drop|--reset")
