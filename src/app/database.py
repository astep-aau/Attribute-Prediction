from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
import os
from dotenv import load_dotenv

# Only load .env file in local development (not in cluster/docker)
if os.getenv("ENV") != "cluster":
    load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("The DATABASE_URL environment variable is not set. Please set it in your environment or .env file.")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

engine = create_async_engine(
    DATABASE_URL,
    echo=DEBUG,
    future=True,
    pool_size=10,
    max_overflow=20
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
