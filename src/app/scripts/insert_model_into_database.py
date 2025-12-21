import sys
import csv
import datetime
import pickle
import os
from pathlib import Path
import time

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.app.services.metric_utils import (
    create_metric,
    create_hyperparam,
    create_loss,
)
from src.app.services.model_type_utils import (
    get_model_type_by_name,
    create_model_type
)
from src.app.services.impute_result_utils import (
    create_impute_result
)
from src.app.schemas import (
    ModelTypeCreate,
    ModelMetricsCreate,
    Hyperparam,
    ModelLoss,
    ImputeResultCreate
)
from src.app.database import AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import json
from uuid import UUID



JSON_FILE_NAME = "GAT_L1_LR0.0001_GNN300_GRU300_H2_D0.2_results.json"
IMPUTE_RESULT_FILE_NAME = "GAT_L1_LR0.0001_GNN300_GRU300_H2_D0.2.csv"
CACHE_PATH = "imputed_lookup.pkl"
DATA_PATH = PROJECT_ROOT / "src" / "app" / "scripts" / "data"

async def main():
    # Read files

    json_path = str(DATA_PATH / JSON_FILE_NAME)
    impute_result_path = str(DATA_PATH / IMPUTE_RESULT_FILE_NAME)

    with open(json_path, 'r') as f:
        model_data_json = json.load(f)

    # Get database session
    async with AsyncSessionLocal() as db:
        # Get model_type name (this is under the assumption that Bi-GRU is always the temporal model)
        model_type_name = model_data_json["gnn_model_used"] + "-BiGRU"

        # Get model_type id
        type_id = await find_or_create_model_type_id(model_type_name, db)

        # Insert model metric into database
        model_metric = await insert_model_metric(model_data_json, type_id, db)

        # Insert hyperparams into database
        await insert_hyperparam(model_data_json, model_metric.id, db)

        # Insert loss into database
        await insert_loss(model_data_json, model_metric.id, db)

        # Insert imputation data
        await insert_imputation(impute_result_path, model_metric.id, db)


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
        path_to_save= f"src/app/scripts/saved_models/{model_data["model_name"] + ".pth"}"
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

        # Removes prefix text e.g.,: test_mape into mape
        loss_unit = str(key).split("_")[-1]
        new_loss = ModelLoss(
            model_id= metric_id,
            type= "test",
            loss_unit= loss_unit,
            loss_value= value
        )
        await create_loss(new_loss, db)

    # Add losses from best_epoch_metrics
    for key, value in model_data["Best_epoch_metrics"].items():
        if ("train" in key):
            loss_type = "train"
        elif ("val" in key):
            loss_type = "validation"

        # Removes prefix text e.g.,: test_mape into mape
        loss_unit = str(key).split("_")[-1]
        new_loss = ModelLoss(
            model_id= metric_id,
            type= loss_type,
            loss_unit= loss_unit,
            loss_value= value
        )
        await create_loss(new_loss, db)

async def insert_imputation(impute_result_path: str, metric_id: UUID, db: AsyncSession):
    start_time = time.time()
    impute_lookup_set = get_impute_set()

    with open(impute_result_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row_idx, row in enumerate(reader, start=1):
            clock = row[0]
            hour, minutes = clock.split(':', 1)
            unix = int(datetime.datetime(2014, 1, 2, int(hour), int(minutes)).timestamp())
            for index, value in enumerate(row[1:], start=1):
                road_id = header[index]
                create_object = ImputeResultCreate(
                    model_id=metric_id,
                    road_id=road_id,
                    tms=unix,
                    value=value,
                    imputed= (unix, road_id) in impute_lookup_set
                )
                await create_impute_result(create_object, db, False)
            if row_idx % 10 == 0:
                print(f"Processed {row_idx} rows")
            await db.commit()

        await db.commit()
        print(f"{"="*60}\n\nTotal processing time for data: {int((time.time() - start_time)/60)}min\n\n{"="*60}")

def get_impute_set():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    imputed_lookup = set()
    path = str(PROJECT_ROOT / "data"/ "edge_data_day7.csv")

    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            clock = row[0]
            hour, minutes = clock.split(":", 1)
            unix = datetime.datetime(2014, 1, 2, int(hour), int(minutes)).timestamp()

            for idx, value in enumerate(row[1:], start=1):
                if value == "-1.0":
                    imputed_lookup.add((int(unix), header[idx]))

    # save cache
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(imputed_lookup, f, protocol=pickle.HIGHEST_PROTOCOL)

    return imputed_lookup

if __name__ == "__main__":
    asyncio.run(main())
    # 2nd jan 2014
