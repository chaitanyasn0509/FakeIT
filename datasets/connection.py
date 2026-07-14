from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///fakeit.db"
# Later you can switch to PostgreSQL:
# DATABASE_URL = "postgresql://user:password@localhost/fakeit"

engine = create_engine(DATABASE_URL, echo=False)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

Base = declarative_base()