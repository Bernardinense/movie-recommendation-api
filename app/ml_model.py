"""
Gerenciamento do modelo de Machine Learning
"""
import joblib
import pandas as pd
import os
from typing import List, Dict, Tuple

from app.recommenders import HybridRecommender, ItemBasedCollaborativeFiltering, PopularityRecommender, SVDRecommender

class MovieRecommenderService:
    """Serviço de recomendação de filmes"""
    
    def __init__(self):
        """Inicializa o serviço carregando modelo e dados"""
        self.hybrid_system = None
        self.ratings = None
        self.movies = None
        self.carregar_recursos()
    
    def carregar_recursos(self):
        """Carrega modelo treinado e datasets"""
        try:
            print("📥 Carregando modelo...")
            
            # Carregar com unpickler customizado
            import pickle
            import sys
            from app import recommenders
            
            # Adicionar módulo ao sys.modules como __main__
            sys.modules['__main__'].HybridRecommender = recommenders.HybridRecommender
            sys.modules['__main__'].ItemBasedCollaborativeFiltering = recommenders.ItemBasedCollaborativeFiltering
            sys.modules['__main__'].PopularityRecommender = recommenders.PopularityRecommender
            sys.modules['__main__'].SVDRecommender = recommenders.SVDRecommender
            
            self.hybrid_system = joblib.load('modelo/hybrid_system.joblib')
            
            print("📥 Carregando ratings...")
            self.ratings = pd.read_csv(
                'ml-100k/u.data',
                sep='\t',
                names=['user_id', 'movie_id', 'rating', 'timestamp']
            )
            
            print("📥 Carregando movies...")
            # Colunas do u.item
            movie_cols = ['movie_id', 'title', 'release_date', 'video_release_date',
                        'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                        'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                        'Thriller', 'War', 'Western']
            
            self.movies = pd.read_csv(
                'ml-100k/u.item',
                sep='|',
                encoding='latin-1',
                names=movie_cols,
                header=None
            )
            
            print("✅ Recursos carregados com sucesso!")
            
        except Exception as e:
            print(f"❌ Erro ao carregar recursos: {e}")
            raise
    
    def recomendar(self, user_id: int, n: int = 5, method: str = 'auto') -> Tuple[List[Dict], str]:
        """
        Gera recomendações para um usuário
        
        Args:
            user_id: ID do usuário
            n: Número de recomendações
            method: Método ('auto', 'collaborative', 'popularity', 'svd')
            
        Returns:
            Tuple com (lista de recomendações, método usado)
        """
        try:
            # Chamar o modelo híbrido
            recomendacoes_df = self.hybrid_system.recommend(
                user_id=user_id,
                ratings=self.ratings,
                movies=self.movies,
                n=n,
                method=method
            )
            
            # Descobrir qual método foi usado
            num_ratings = len(self.ratings[self.ratings['user_id'] == user_id])
            if method == 'auto':
                method_used = 'collaborative' if num_ratings >= 5 else 'popularity'
            else:
                method_used = method
            
            # Converter DataFrame para lista de dicts
            recomendacoes = []
            for _, row in recomendacoes_df.iterrows():
                recomendacoes.append({
                    'movie_id': int(row['movie_id']),
                    'title': str(row['title']),
                    'score': float(row.get('score', row.get('predicted_rating', 0.0)))
                })
            
            return recomendacoes, method_used
            
        except Exception as e:
            print(f"❌ Erro ao gerar recomendações: {e}")
            raise

# Instância global (carregada uma vez quando a API inicia)
recommender_service = MovieRecommenderService()