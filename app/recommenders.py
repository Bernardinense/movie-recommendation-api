
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class PopularityRecommender:
    """Recomendador baseado em popularidade"""
    
    def __init__(self, min_ratings=50):
        self.min_ratings = min_ratings
        self.popular_movies = None
    
    def fit(self, ratings):
        movie_stats = ratings.groupby('movie_id').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        movie_stats.columns = ['movie_id', 'count', 'mean_rating']
        
        popular = movie_stats[movie_stats['count'] >= self.min_ratings]
        popular = popular.sort_values('mean_rating', ascending=False)
        
        self.popular_movies = popular
        return self
    
    def recommend(self, user_id, n=5):
        return self.popular_movies.head(n)


class ItemBasedCollaborativeFiltering:
    """Filtragem colaborativa baseada em itens"""
    
    def __init__(self, k=20):
        self.k = k
        self.similarity_matrix = None
        self.user_movie_matrix = None
    
    def fit(self, ratings):
        self.user_movie_matrix = ratings.pivot(
            index='user_id',
            columns='movie_id',
            values='rating'
        ).fillna(0)
        
        self.similarity_matrix = cosine_similarity(self.user_movie_matrix.T)
        self.similarity_matrix = pd.DataFrame(
            self.similarity_matrix,
            index=self.user_movie_matrix.columns,
            columns=self.user_movie_matrix.columns
        )
        
        return self
    
    def get_similar_movies(self, movie_id, n=10):
        if movie_id not in self.similarity_matrix.index:
            return pd.Series()
        
        similar = self.similarity_matrix[movie_id].sort_values(ascending=False)
        return similar.iloc[1:n+1]
    
    def recommend(self, user_id, ratings, movies, n=5):
        user_ratings = ratings[ratings['user_id'] == user_id]
        
        if len(user_ratings) == 0:
            return pd.DataFrame()
        
        user_rated_movies = set(user_ratings['movie_id'].values)
        
        candidates = {}
        for _, row in user_ratings.iterrows():
            movie_id = row['movie_id']
            rating = row['rating']
            
            similar_movies = self.get_similar_movies(movie_id, self.k)
            
            for sim_movie_id, similarity in similar_movies.items():
                if sim_movie_id not in user_rated_movies:
                    if sim_movie_id not in candidates:
                        candidates[sim_movie_id] = []
                    candidates[sim_movie_id].append(rating * similarity)
        
        recommendations = []
        for movie_id, scores in candidates.items():
            avg_score = np.mean(scores)
            recommendations.append({
                'movie_id': movie_id,
                'score': avg_score
            })
        
        recommendations = pd.DataFrame(recommendations)
        recommendations = recommendations.sort_values('score', ascending=False)
        recommendations = recommendations.head(n)
        
        recommendations = recommendations.merge(
            movies[['movie_id', 'title']],
            on='movie_id',
            how='left'
        )
        
        return recommendations


class SVDRecommender:
    """Recomendador usando SVD"""
    
    def __init__(self, n_factors=50, random_state=42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        self.user_movie_matrix = None
        self.all_movie_ids = None
        self.svd = None
    
    def fit(self, ratings):
        from sklearn.decomposition import TruncatedSVD
        
        self.user_movie_matrix = ratings.pivot(
            index='user_id',
            columns='movie_id',
            values='rating'
        ).fillna(0)
        
        self.all_movie_ids = self.user_movie_matrix.columns.tolist()
        self.global_mean = ratings['rating'].mean()
        
        self.svd = TruncatedSVD(
            n_components=self.n_factors,
            random_state=self.random_state
        )
        
        self.user_factors = self.svd.fit_transform(self.user_movie_matrix)
        self.item_factors = self.svd.components_.T
        
        return self
    
    def predict(self, user_id, movie_id):
        try:
            user_idx = list(self.user_movie_matrix.index).index(user_id)
            movie_idx = self.all_movie_ids.index(movie_id)
            
            prediction = np.dot(
                self.user_factors[user_idx],
                self.item_factors[movie_idx]
            )
            
            return max(1, min(5, prediction))
        except (ValueError, IndexError):
            return self.global_mean
    
    def recommend(self, user_id, ratings, movies, n=5):
        user_ratings = ratings[ratings['user_id'] == user_id]
        user_rated_movies = set(user_ratings['movie_id'].values)
        
        predictions = []
        for movie_id in self.all_movie_ids:
            if movie_id not in user_rated_movies:
                pred = self.predict(user_id, movie_id)
                predictions.append({
                    'movie_id': movie_id,
                    'predicted_rating': pred
                })
        
        recommendations = pd.DataFrame(predictions)
        recommendations = recommendations.sort_values(
            'predicted_rating',
            ascending=False
        ).head(n)
        
        recommendations = recommendations.merge(
            movies[['movie_id', 'title']],
            on='movie_id',
            how='left'
        )
        
        return recommendations


class HybridRecommender:
    """Sistema híbrido de recomendação"""
    
    def __init__(self, svd_model, collaborative_model, popularity_model):
        self.svd = svd_model
        self.collaborative = collaborative_model
        self.popularity = popularity_model
    
    def recommend(self, user_id, ratings, movies, n=5, method='auto'):
        num_ratings = len(ratings[ratings['user_id'] == user_id])
        
        if method == 'auto':
            if num_ratings >= 5:
                method = 'collaborative'
            else:
                method = 'popularity'
        
        if method == 'collaborative':
            return self.collaborative.recommend(user_id, ratings, movies, n)
        elif method == 'popularity':
            result = self.popularity.recommend(user_id, n)
            return result.merge(
                movies[['movie_id', 'title']],
                on='movie_id',
                how='left'
            )
        elif method == 'svd':
            return self.svd.recommend(user_id, ratings, movies, n)
        else:
            raise ValueError(f"Método '{method}' não reconhecido")