import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from http import HTTPStatus
from typing import Dict, List, Union
from sklearn.linear_model import LinearRegression
import joblib
import os

app = FastAPI(
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)

# Модели данных
class FitRequest(BaseModel):
    config: Dict[str, str]
    train_data: Dict[str, List[Union[List[float], float]]]

class LoadRequest(BaseModel):
    model_ids: List[str]

class PredictRequest(BaseModel):
    model_id: str
    features: List[List[float]]

class ApiResponse(BaseModel):
    message: str
    data: Union[Dict, None] = None

class ModelListResponse(BaseModel):
    models: List[str]

# Хранилище моделей
models_db: Dict[str, str] = {}  # Хранит путь к сохраненным моделям
loaded_models: Dict[str, LinearRegression] = {}  # Загруженные модели в памяти

# Endpoints
@app.post("/fit", response_model=ApiResponse, status_code=HTTPStatus.CREATED)
async def fit(request: FitRequest):
    try:
        # Данные для обучения
        X = [pair[0] for pair in request.train_data.values()]
        y = [pair[1] for pair in request.train_data.values()]
        model_id = request.config.get("model_id", "default_model")

        # Создание и обучение модели
        model = LinearRegression()
        model.fit(X, y)

        # Сохранение модели
        model_path = f"{model_id}.joblib"
        joblib.dump(model, model_path)
        models_db[model_id] = model_path

        return ApiResponse(message=f"Model '{model_id}' trained and saved", data={"model_id": model_id})
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail={"message": str(e)}
        )

@app.post("/load", response_model=ApiResponse)
async def load(request: LoadRequest):
    try:
        for model_id in request.model_ids:
            if model_id not in models_db:
                raise HTTPException(
                    status_code=HTTPStatus.NOT_FOUND,
                    detail={"message": f"Model '{model_id}' not found"}
                )
            model_path = models_db[model_id]
            loaded_models[model_id] = joblib.load(model_path)
        return ApiResponse(message="Models loaded successfully")
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail={"message": str(e)}
        )

@app.post("/predict", response_model=ApiResponse)
async def predict(request: PredictRequest):
    try:
        model = loaded_models.get(request.model_id)
        if model is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail={"message": f"Model '{request.model_id}' not loaded"}
            )
        predictions = model.predict(request.features).tolist()
        return ApiResponse(message="Prediction successful", data={"predictions": predictions})
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail={"message": str(e)}
        )

@app.get("/list_models", response_model=ModelListResponse)
async def list_models():
    return ModelListResponse(models=list(models_db.keys()))

@app.delete("/remove_all", response_model=ApiResponse)
async def remove_all():
    try:
        for model_id, model_path in models_db.items():
            if os.path.exists(model_path):
                os.remove(model_path)
        models_db.clear()
        loaded_models.clear()
        return ApiResponse(message="All models removed successfully")
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail={"message": str(e)}
        )
