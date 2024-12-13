import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, root_validator, ValidationError
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
    fit_intercept: bool = True

class Config(BaseModel):
    hyperparameters: Hyperparameters
    id: str = "model"

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

@app.post("/fit", response_model=List[SuccessResponse], status_code=HTTPStatus.CREATED)
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

@app.post("/load", response_model=List[SuccessResponse], responses={422: {"model": ErrorResponse}})
async def load(file: Load):
    try:
        responses = []
        model = joblib.load(f'{file.id}.joblib')
        loaded_models[file.id] = model
        print("fit: ",loaded_models)
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
    
@app.post("/predict", response_model=List[SuccessPrediction], responses={422: {"model": ErrorResponse}})
async def predict(predict_model: Predict):
    try:
        responses = []
        print(predict_model)
        model = loaded_models.get(predict_model.id)
        predictions = model.predict(predict_model.X).tolist()
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
    
@app.get("/list_models", response_model=List[Model])
async def list_models():
    return models_db.models

@app.delete("/remove_all", response_model=List[SuccessResponse])
async def remove_all():
    responses = []
    for model in models_db.models:
        print(loaded_models)
        responses.append(SuccessResponse(message=f"Model '{model.id}' removed"))
        os.remove(f'{model.id}.joblib')
        if model.id in loaded_models:
            del loaded_models[model.id]
    models_db.models = []
    return responses

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    