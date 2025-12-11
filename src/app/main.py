from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from src.app.routes import metric_routes, model_type_routes, impute_results_routes, download_model_routes
from src.app.database import engine, Base
from src.app.exceptions import NotFoundException, InvalidUUIDException
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

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": f"Invalid input: {str(exc)}"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    # Log the error for debugging (you can add proper logging here)
    print(f"Unexpected error: {type(exc).__name__}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

app.include_router(model_type_routes.router)
app.include_router(metric_routes.router)
app.include_router(impute_results_routes.router)
app.include_router(download_model_routes.router)

# uvicorn src.app.main:app --reload
