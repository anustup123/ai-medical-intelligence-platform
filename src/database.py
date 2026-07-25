"""
Database layer using SQLAlchemy ORM.

Default: SQLite (zero-config, file-based — perfect for an assignment/demo).
Production note: Only DATABASE_URL needs to change to point this at
PostgreSQL/MySQL, e.g.:
    DATABASE_URL=postgresql://user:password@host:5432/mediai
No other code changes are required — that's the point of using an ORM.
"""
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String(255))
    predicted_class = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    probability_normal = Column(Float)
    probability_pneumonia = Column(Float)
    attention_region = Column(String(100))
    gradcam_image_path = Column(String(500))
    llm_report = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
