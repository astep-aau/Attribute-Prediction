from fastapi import FastAPI
from src.app.routes import metric_routes, model_type_routes, impute_results_routes

app = FastAPI()

app.include_router(model_type_routes.router)
app.include_router(metric_routes.router)
app.include_router(impute_results_routes.router)
