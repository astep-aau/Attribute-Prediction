from fastapi import FastAPI
from src.app.routes import metric_routes, model_type_routes, impute_results_routes, download_model_routes
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.app.database import get_db, engine, Base
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Shutdown: Close connections
    await engine.dispose()

app = FastAPI(lifespan=lifespan)

app.include_router(model_type_routes.router)
app.include_router(metric_routes.router)
app.include_router(impute_results_routes.router)
app.include_router(download_model_routes.router)

# uvicorn src.app.main:app --reload
