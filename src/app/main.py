from fastapi import FastAPI
from src.app.routes import model_routes, metric_routes

app = FastAPI()

app.include_router(model_routes.router)
app.include_router(metric_routes.router)
