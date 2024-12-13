import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from http import HTTPStatus
from sklearn.linear_model import LinearRegression
import joblib
import os

app = FastAPI(
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)

class Hyperparameters(BaseModel):
    fit_intercept: bool

class Config(BaseModel):
    hyperparameters: Hyperparameters
    id: str

class TrainingData(BaseModel):
    X: List[List[float]] = [[1,2], [3,4]]
    config: Config 
    y: List[float] = [5,6] 

class SuccessResponse(BaseModel):
    message: str

class ErrorResponse(BaseModel):
    loc: List[str]
    msg: str
    type: str

class Load(BaseModel):
    id: str

class Predict(BaseModel):
    id: str
    X: List[List[float]] = [[0]]

class SuccessPrediction(BaseModel):
    predictions: List[float]

class Model(BaseModel):
    id: str

class Models(BaseModel):
    models: List[Model] = []

models_db = Models()
loaded_models: Dict[str, LinearRegression] = {}

@app.post("/api/v1/models/fit", response_model=List[SuccessResponse], status_code=HTTPStatus.CREATED)
async def fit(training_data: List[TrainingData]):
    try:
        responses = []
        for data in training_data:
            X = data.X
            y = data.y
            config = data.config
            fit_intercept = config.hyperparameters.fit_intercept
            existing_model = next((model for model in models_db.models if model.id == config.id), None)

            model = LinearRegression(fit_intercept=fit_intercept)
            model.fit(X, y)
            model_path = f"{config.id}.joblib"
            joblib.dump(model, model_path)
            if existing_model == None:
                models_db.models.append(Model(id=config.id))
            responses.append(SuccessResponse(message=f"Model '{config.id}' trained and saved"))
        return responses
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail=[{
                "loc": ["fit"],
                "msg": str(e),
                "type": "error"
            }]
        )

@app.post("/api/v1/models/load", response_model=List[SuccessResponse], responses={422: {"model": ErrorResponse}})
async def load(load_model_id: List[Load]):
    try:
        responses = []
        for file in load_model_id:
            model = joblib.load(f'{file.id}.joblib')
            loaded_models[file.id] = model
            responses.append(SuccessResponse(message=f"Model '{file.id}' loaded"))
        return responses
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=[{
                "loc": ["load"],
                "msg": str(e),
                "type": "error"
            }]
        )
    
@app.post("/api/v1/models/predict", response_model=List[SuccessPrediction], responses={422: {"model": ErrorResponse}})
async def predict(predict_model: List[Predict]):
    try:
        responses = []
        for pred in predict_model:
            model = loaded_models.get(pred.id)
            predictions = model.predict(pred.X).tolist()
            responses.append(SuccessPrediction(predictions=predictions))
        return responses
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=[{
                "loc": ["predict"],
                "msg": str(e),
                "type": "error"
            }]
        )
    
@app.get("/api/v1/models/list_models", response_model=Models)
async def list_models():
    return models_db

@app.delete("/api/v1/models/remove_all", response_model=List[SuccessResponse])
async def remove_all():
    responses = []
    for model in models_db.models:
        responses.append(SuccessResponse(message=f"Model '{model.id}' removed"))
        os.remove(f'{model.id}.joblib')
        del loaded_models[model.id]
    models_db.models = []
    return responses

