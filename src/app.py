from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import joblib
import pandas as pd
from fastapi import FastAPI, Request

from src.schemas import PredictionRequest, PredictionResponse

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pipeline_rede_neural.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.modelo = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="API de Predição de Sobrevivência",
    description="Inferência do pipeline MLP treinado com o dataset Titanic.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/diagnostico/modelo")
async def diagnostico_modelo() -> dict[str, str]:
    return {
        "status": "operacional_v2",
        "framework": "FastAPI-Production",
        "modelo": "MLPClassifier",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    dados = pd.DataFrame([payload.model_dump()]).rename(
        columns={
            "pclass": "Pclass",
            "sex": "Sex",
            "age": "Age",
            "sib_sp": "SibSp",
            "parch": "Parch",
            "fare": "Fare",
            "embarked": "Embarked",
        }
    )
    modelo = request.app.state.modelo
    categoria = int(modelo.predict(dados)[0])
    probabilidades = modelo.predict_proba(dados)[0]

    return PredictionResponse(
        categoria_prevista=categoria,
        score_confianca=float(probabilidades.max()),
    )