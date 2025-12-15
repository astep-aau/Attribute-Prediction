from fastapi import APIRouter, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from src.app.database import get_db

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Basic health check endpoint

    Returns:
        Status message indicating the service is running
    """
    return {"status": "healthy", "service": "Attribute-Prediction API"}

@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_probe():
    """
    Kubernetes liveness probe - checks if app should be restarted

    Returns:
        Simple status indicating application is alive
    """
    return {"status": "alive"}

@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_probe(db: AsyncSession = Depends(get_db)):
    """
    Kubernetes readiness probe - checks if app can accept traffic
    Verifies database connectivity before marking pod as ready

    Args:
        db: Database session

    Returns:
        Status message including database connectivity status

    Raises:
        Exception: If database connection fails (returns 500, pod marked not ready)
    """
    try:
        # Simple query to check database connectivity
        await db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "service": "Attribute-Prediction API",
            "database": "connected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Attribute-Prediction API",
            "database": "disconnected",
            "error": str(e)
        }
