from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from src.app.routes import metric_routes, model_type_routes, impute_results_routes, download_model_routes, health_routes
from src.app.database import engine, Base
from src.app.exceptions import NotFoundException, InvalidUUIDException, ForeignKeyViolationException
from contextlib import asynccontextmanager
import logging
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO if os.getenv("ENV") == "cluster" else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create tables if they don't exist
    logger.info("Application startup - initializing database tables")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized successfully")
    yield
    # Shutdown: Close connections
    logger.info("Application shutdown - closing database connections")
    await engine.dispose()
    logger.info("Database connections closed")

app = FastAPI(lifespan=lifespan)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url.path}")

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(f"Response: {response.status_code} | Duration: {duration:.3f}s")

    return response

# Exception handlers
@app.exception_handler(NotFoundException)
async def not_found_handler(request: Request, exc: NotFoundException):
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc) if str(exc) else "Resource not found"}
    )

@app.exception_handler(InvalidUUIDException)
async def invalid_uuid_handler(request: Request, exc: InvalidUUIDException):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc) if str(exc) else "Invalid UUID format"}
    )

@app.exception_handler(ForeignKeyViolationException)
async def foreign_key_violation_handler(request: Request, exc: ForeignKeyViolationException):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": f"Invalid input: {str(exc)}"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    # Log the error for debugging in cluster logs
    logger.error(f"Unexpected error: {type(exc).__name__}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

app.include_router(model_type_routes.router)
app.include_router(metric_routes.router)
app.include_router(impute_results_routes.router)
app.include_router(download_model_routes.router)
app.include_router(health_routes.router)

# uvicorn src.app.main:app --reload
