"""
Schemas Pydantic para validação de dados da API
"""
from pydantic import BaseModel, Field
from typing import List, Optional

class RecomendacaoInput(BaseModel):
    """Entrada para solicitar recomendações"""
    user_id: int = Field(..., description="ID do usuário", ge=1)
    n: int = Field(5, description="Número de recomendações", ge=1, le=20)
    method: str = Field('auto', description="Método: 'auto', 'collaborative', 'popularity', 'svd'")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 196,
                "n": 5,
                "method": "auto"
            }
        }

class FilmeRecomendado(BaseModel):
    """Informação de um filme recomendado"""
    movie_id: int
    title: str
    score: float

class RecomendacaoResponse(BaseModel):
    """Resposta da API com recomendações"""
    user_id: int
    method_used: str
    recommendations: List[FilmeRecomendado]
    total: int