"""
API FastAPI para Sistema de Recomendação de Filmes
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.models import RecomendacaoInput, RecomendacaoResponse, FilmeRecomendado
from app.ml_model import recommender_service
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== LIFESPAN ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia startup e shutdown da API"""
    # Startup
    logger.info("=" * 60)
    logger.info("🚀 Iniciando API de Recomendação de Filmes...")
    logger.info("📊 Sistema Híbrido (Incorporado do Dia 4) integrado!")
    logger.info("=" * 60)
    yield
    # Shutdown
    logger.info("👋 API encerrada!")

# Criar aplicação FastAPI
app = FastAPI(
    title="🎬 API de Recomendação de Filmes - MovieLens",
    description="API para recomendar filmes usando sistema híbrido de Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan 
)

# CORS (permitir requisições de outros domínios)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== ENDPOINTS ====================

@app.get("/", tags=["Geral"])
async def root():
    """Endpoint raiz"""
    return {
        "mensagem": "🎬 API de Recomendação de Filmes",
        "versao": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "recomendar": "/recomendar",
            "stats": "/stats"
        }
    }

@app.get("/health", tags=["Health Check"])
async def health_check():
    """Health check - Verifica se a API está funcionando"""
    try:
        # Verificar se o serviço está OK
        if recommender_service.hybrid_system is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modelo não carregado"
            )
        
        total_ratings = len(recommender_service.ratings)
        total_movies = len(recommender_service.movies)
        
        return {
            "status": "healthy",
            "modelo": "carregado",
            "total_avaliacoes": total_ratings,
            "total_filmes": total_movies
        }
    
    except Exception as e:
        logger.error(f"Erro no health check: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Serviço indisponível: {str(e)}"
        )

@app.post("/recomendar",
          response_model=RecomendacaoResponse,
          tags=["Recomendação"],
          status_code=status.HTTP_200_OK)
async def recomendar_filmes(entrada: RecomendacaoInput) -> RecomendacaoResponse:
    """
    **Endpoint principal**: Recebe ID de usuário e retorna recomendações
    
    - **user_id**: ID do usuário (1-943)
    - **n**: Número de recomendações (1-20, padrão: 5)
    - **method**: Método de recomendação ('auto', 'collaborative', 'popularity', 'svd')
    
    **Lógica 'auto':**
    - Se usuário tem ≥5 avaliações → Collaborative Filtering
    - Se usuário tem <5 avaliações → Popularity-based
    """
    try:
        logger.info(f"📥 Requisição: user_id={entrada.user_id}, n={entrada.n}, method={entrada.method}")
        
        # Validar se user_id existe
        user_exists = entrada.user_id in recommender_service.ratings['user_id'].values
        if not user_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Usuário {entrada.user_id} não encontrado no dataset"
            )
        
        # Gerar recomendações
        recomendacoes_raw, method_used = recommender_service.recomendar(
            user_id=entrada.user_id,
            n=entrada.n,
            method=entrada.method
        )
        
        # Converter para Pydantic models
        recomendacoes = [
            FilmeRecomendado(**rec) for rec in recomendacoes_raw
        ]
        
        # Preparar resposta
        resposta = RecomendacaoResponse(
            user_id=entrada.user_id,
            method_used=method_used,
            recommendations=recomendacoes,
            total=len(recomendacoes)
        )
        
        logger.info(f"✅ {len(recomendacoes)} recomendações geradas usando '{method_used}'")
        
        return resposta
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"❌ Erro ao gerar recomendações: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno: {str(e)}"
        )

@app.get("/stats", tags=["Estatísticas"])
async def estatisticas():
    """Retorna estatísticas do dataset"""
    try:
        ratings = recommender_service.ratings
        movies = recommender_service.movies
        
        return {
            "total_usuarios": int(ratings['user_id'].nunique()),
            "total_filmes": len(movies),
            "total_avaliacoes": len(ratings),
            "media_avaliacoes_por_usuario": float(ratings.groupby('user_id').size().mean()),
            "rating_medio": float(ratings['rating'].mean())
        }
    
    except Exception as e:
        logger.error(f"❌ Erro ao gerar estatísticas: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao gerar estatísticas: {str(e)}"
        )