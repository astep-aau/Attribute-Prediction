from fastapi import FastAPI
from src.app.routes import model_routes

app = FastAPI()

app.include_router(model_routes.router)
