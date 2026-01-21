# AI Agent Instructions - Attribute Prediction System

## Project Overview

Dual-component system for road traffic attribute prediction: **ML training pipeline** (PyTorch/GNN models) and **FastAPI microservice** (production API). The project manages separate concerns: data scientists work in `src/models/` and `src/data_manipulation/`, while API developers work in `src/app/`.

## Architecture & Data Flow

### ML Pipeline (`src/models/`, `src/data_manipulation/`)
- **Entry points**: [`src/models/main_train.py`](../src/models/main_train.py) for training, [`src/main_imputation.py`](../src/main_imputation.py) for inference
- **Data flow**: `FileLoader` → `StaticGraphBuilder` (builds graph structure) → `GraphDatasetBuilder` (creates PyTorch Geometric datasets) → `DataLoader` batches
- **Models**: GAT_BiGRU and GraphSAGE_BiGRU imputers in [`src/models/`](../src/models/) - graph neural networks for traffic data imputation
- **Global edge consistency**: All datasets must use `get_global_edge_columns_and_ids()` to ensure uniform node count (N) across train/val/test splits

### API Microservice (`src/app/`)
- **Architecture**: FastAPI with async SQLAlchemy 2.0, layered as routes → services → database_tables (ORM)
- **Database**: PostgreSQL with `models` schema, accessed via SSH tunnel in local dev: `ssh -L 5432:cs-astep02.srv.aau.dk:30432 username@student.aau.dk -N`
- **Responsibilities**: Serve model metadata, metrics (training time, bias, gap), imputation results, and `.pth` model file downloads
- **Key pattern**: UUIDs for all primary keys, foreign key relationships between `model_type` → `model_metrics` → `hyperparams`/`loss` tables

## Critical Developer Workflows

### Running the API Locally
```bash
# 1. Start SSH tunnel (keep running in separate terminal)
ssh -L 5432:cs-astep02.srv.aau.dk:30432 username@student.aau.dk -N

# 2. Configure .env with tunnel connection
DATABASE_URL=postgresql+asyncpg://username:password@localhost:5432/attribute_prediction

# 3. Run server
uvicorn src.app.main:app --reload
```

### Testing
```bash
# Run microservice tests (NOT ML tests)
pytest tests/mircro_service/ -v

# With coverage
pytest tests/mircro_service/ --cov=src/app --cov-report=html
```
**Test structure**: [`tests/mircro_service/conftest.py`](../tests/mircro_service/conftest.py) provides in-memory SQLite fixtures. Follow AAA pattern (Arrange-Act-Assert) and naming: `test_<function>_<scenario>`.

### Training Models
```bash
cd src/models
python main_train.py  # Configure hyperparameters in config.py
```
**Config source**: [`src/models/config.py`](../src/models/config.py) sets `SEQ_LEN=12`, `BATCH_SIZE`, `MASK_RATE=0.2`, GNN/GRU dimensions, etc.

### Running Imputation Inference
```bash
python src/main_imputation.py
```
Loads checkpoints from `src/trained_models/pth/*.pth`, parses hyperparams from filename regex (e.g., `GAT_L1_LR0.0001_GNN200_GRU200_H1_D0.2.pth`).

## Project-Specific Conventions

### FastAPI Service Layer Pattern
- **Routes** ([`src/app/routes/`](../src/app/routes/)): Define endpoints, validate request schemas (Pydantic), handle HTTP concerns
- **Services** ([`src/app/services/`](../src/app/services/)): Business logic, database queries, exception raising (`NotFoundException`, `ForeignKeyViolationException`)
- **Database Tables** ([`src/app/database_tables/`](../src/app/database_tables/)): SQLAlchemy ORM models with `__table_args__ = {"schema": "models"}`
- **Never** put database logic in routes; always delegate to service functions

### Database Schema Conventions
- All tables use `UUID` primary keys: `id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)`
- Foreign keys reference full path: `ForeignKey("models.model_type.id")`
- Timestamps use `server_default=func.now()` for creation time
- Service functions validate UUIDs and raise `InvalidUUIDException` for malformed strings

### ML Data Pipeline Pattern
1. **FileLoader** reads CSV traffic data with columns like `edge123_avg_speed_km_h_sec`
2. **StaticGraphBuilder** creates graph structure from `edge_connections.csv` (road network topology) and OSM metadata
3. **GraphDatasetBuilder** produces PyTorch Geometric `Data` objects with node features, edge indices, and masks
4. Always use `global_edge_ids` to ensure consistent node ordering across datasets

### Model Checkpoint Naming
Filenames encode hyperparameters: `{MODEL}_L{layers}_LR{lr}_GNN{gnn_dim}_GRU{gru_dim}_H{heads}_D{dropout}.pth`
- Example: `GAT_L1_LR0.0001_GNN200_GRU200_H2_D0.2.pth`
- Parsed by regex in [`main_imputation.py`](../src/main_imputation.py) for inference configuration

## Deployment & CI/CD

### Kubernetes Deployment
- **Namespace**: `cs-25-sw-5-06`
- **Image**: `ghcr.io/astep-aau/attribute-prediction:latest`
- **Workflow**: [`.github/workflows/deploy.yml`](../../../.github/workflows/deploy.yml) auto-deploys on push to `main` (build → push → rollout restart)
- **Health checks**: `/health/live` (liveness probe), `/health/ready` (readiness with DB validation)
- **Environment**: `ENV=cluster` disables `.env` loading, uses Kubernetes secrets for `DATABASE_URL`

### Dockerfile Context
[`Dockerfile`](../../Dockerfile) uses Python 3.13, copies `requirements.txt` first (layer caching), runs `uvicorn src.app.main:app` on port 8000.

## Key Files Reference

- **API entry**: [`src/app/main.py`](../src/app/main.py) - FastAPI app with exception handlers, middleware, lifespan events
- **Database setup**: [`src/app/database.py`](../src/app/database.py) - Async engine, session factory, pool configuration
- **ML config**: [`src/models/config.py`](../src/models/config.py) - Central hyperparameter definitions
- **Test fixtures**: [`tests/mircro_service/conftest.py`](../tests/mircro_service/conftest.py) - SQLite in-memory DB for isolated tests
- **K8s manifests**: [`k8s/deployment.yaml`](../k8s/deployment.yaml), [`k8s/service.yaml`](../k8s/service.yaml)

## Common Pitfalls

- **Local dev without SSH tunnel**: API database connections will fail; must forward port 5432 to remote server
- **Testing ML code**: Run `pytest tests/mircro_service/` NOT root `tests/` (ML tests incomplete)
- **Schema mismatch in tests**: [`conftest.py`](../tests/mircro_service/conftest.py) removes `schema="models"` for SQLite compatibility
- **Missing global_edge_ids**: Training/inference must use same edge set or node indices shift, breaking model loading
- **Foreign key violations**: Ensure `model_type` exists before creating `model_metrics`; service layer validates this

## Documentation Locations

- **API details**: [`src/app/README.md`](../src/app/README.md) - Setup, endpoints, deployment
- **ML info**: [`ML_README.md`](../ML_README.md) - (Managed by ML team, incomplete)
- **Test guidelines**: [`tests/best_pratices.md`](../tests/best_pratices.md) - AAA pattern, fixtures, parametrize
