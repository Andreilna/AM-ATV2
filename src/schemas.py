from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pclass: int = Field(ge=1, le=3, description="Classe da passagem (1, 2 ou 3)", examples=[1])
    sex: Literal["female", "male"] = Field(description="Sexo registrado do passageiro", examples=["female"])
    age: float = Field(ge=0, le=100, description="Idade em anos", examples=[35])
    sib_sp: int = Field(ge=0, le=8, description="Número de irmãos/cônjuges a bordo", examples=[1])
    parch: int = Field(ge=0, le=6, description="Número de pais/filhos a bordo", examples=[0])
    fare: float = Field(ge=0, le=600, description="Tarifa paga em libras", examples=[53.1])
    embarked: Literal["C", "Q", "S"] = Field(
        description="Porto de embarque: Cherbourg (C), Queenstown (Q) ou Southampton (S)",
        examples=["S"],
    )


class PredictionResponse(BaseModel):
    categoria_prevista: int = Field(description="Classe prevista: 0 (não sobreviveu) ou 1 (sobreviveu)")
    score_confianca: float = Field(ge=0.0, le=1.0, description="Maior probabilidade estimada pelo modelo")