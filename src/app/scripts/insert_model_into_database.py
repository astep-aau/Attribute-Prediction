import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.app.services.metric_utils import (
    create_metric,
    create_hyperparam,
    create_loss
)
from src.app.services.model_type_utils import (
    get_model_type_by_name,
    create_model_type
)
from src.app.schemas import (
    ModelTypeCreate,
    ModelMetricsCreate,
    Hyperparam,
    ModelLoss
)
from src.app.database import AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import json
from uuid import UUID

JSON_FILE_NAME = "GraphSAGE_HPT15_L0.002_GNN256_GRU256_D0.2_A1_results.json"
IMPUTE_RESULT_FILE_NAME = "GraphSAGE_HPT15_L0.002_GNN256_GRU256_D0.2_A1.csv"

async def main():
    # Read files
    data_path = project_root / "src" / "app" / "scripts" / "data"
    json_path = str(data_path / JSON_FILE_NAME)
    impute_result_path = str(data_path / IMPUTE_RESULT_FILE_NAME)

    with open(json_path, 'r') as f:
        model_data_json = json.load(f)

    # Get database session
    async with AsyncSessionLocal() as db:
        # Get model_type name (this is under the assumtion that Bi-GRU is always the temporal model)
        model_type_name = model_data_json["gnn_model_used"] + "-BiGRU"

        # Get model_type id
        type_id = await find_or_create_model_type_id(model_type_name, db)

        # Insert model metric into database
        model_metric = await insert_model_metric(model_data_json, type_id, db)

        # Insert hyperparams into databse
        await insert_hyperparam(model_data_json, model_metric.id, db)

        # Insert loss into databse
        await insert_loss(model_data_json, model_metric.id, db)

        # TODO Add insertion of imputation data

async def find_or_create_model_type_id(name: str, db: AsyncSession) -> UUID:
    model_type_entry = await get_model_type_by_name(name, db)

    if not model_type_entry:
        new_model_type = ModelTypeCreate(name=name)
        model_type_entry = await create_model_type(new_model_type, db)

    return model_type_entry.id

async def insert_model_metric(model_data: dict, type_id: UUID, db: AsyncSession):
    new_metric = ModelMetricsCreate(
        model_type= type_id,
        train_time_min= int(model_data["timing"]["total_training_time_s"]),
        bias= model_data["metrics"]["bias"],
        gap= model_data["metrics"]["overfitting_gap_val_diff"],
        path_to_save= f"src/app/scripts/saved_models{model_data["model_name"] + ".pth"}"
    )
    model_metric = await create_metric(new_metric, db)
    return model_metric

async def insert_hyperparam(model_data: dict, metric_id: UUID, db: AsyncSession):
    for key, value in model_data["hyperparameters"].items():
        new_hyperparam = Hyperparam(
            model_id= metric_id,
            param_name= key,
            param_value= str(value)
        )
        await create_hyperparam(new_hyperparam, db)

    epoch_param = Hyperparam(
        model_id= metric_id,
        param_name= "epochs",
        param_value= str(model_data["timing"]["epochs_completed"])
    )
    await create_hyperparam(epoch_param, db)

async def insert_loss(model_data: dict, metric_id: UUID,  db: AsyncSession):

    # Add losses from 'metrics' object
    for key, value in model_data["metrics"].items():

        if ("test" not in key):
            continue

        # Removes prefixs texts fx: test_mape into mape
        loss_unit = str(key).split("_")[-1]
        new_loss = ModelLoss(
            model_id= metric_id,
            type= "test",
            loss_unit= loss_unit,
            loss_value= value
        )
        await create_loss(new_loss, db)

    # Add losses from best_epich_metrics
    for key, value in model_data["best_epoch_metrics"].items():
        if ("train" in key):
            loss_type = "test"
        elif ("val" in key):
            loss_type = "validation"

        # Removes prefixs texts fx: test_mape into mape
        loss_unit = str(key).split("_")[-1]
        new_loss = ModelLoss(
            model_id= metric_id,
            type= loss_type,
            loss_unit= loss_unit,
            loss_value= value
        )
        await create_loss(new_loss, db)

if __name__ == "__main__":
    asyncio.run(main())
