import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")

MIN_RATINGS = 25
GENRE_WEIGHT = 0.35
RATING_WEIGHT = 0.65


def load_movies(path: str = MOVIES_PATH) -> pd.DataFrame:
    """Load movie metadata and normalize text fields."""
    movies = pd.read_csv(path)
    if "movieId" not in movies.columns or "title" not in movies.columns or "genres" not in movies.columns:
        raise ValueError("movies.csv must contain 'movieId', 'title', and 'genres' columns")

    movies["genres"] = movies["genres"].fillna("").astype(str)
    return movies


def load_ratings(path: str = RATINGS_PATH) -> pd.DataFrame:
    """Load the raw ratings dataset."""
    ratings = pd.read_csv(path)
    if "userId" not in ratings.columns or "movieId" not in ratings.columns or "rating" not in ratings.columns:
        raise ValueError("ratings.csv must contain 'userId', 'movieId', and 'rating' columns")
    return ratings


def build_movie_stats(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Compute rating counts and average ratings for every movie."""
    stats = (
        ratings.groupby("movieId")["rating"]
        .agg(rating_count="count", average_rating="mean")
        .reset_index()
    )
    return movies.merge(stats, on="movieId", how="left").fillna({"rating_count": 0, "average_rating": 0.0})


def build_genre_matrix(movies: pd.DataFrame):
    """Convert pipe-separated genres into a binary feature matrix."""
    vectorizer = CountVectorizer(token_pattern=r"[^|]+")
    genre_matrix = vectorizer.fit_transform(movies["genres"])
    return genre_matrix, vectorizer


def build_genre_similarity(genre_matrix):
    """Compute similarity between movies based on genre overlap."""
    return cosine_similarity(genre_matrix, genre_matrix)


def build_rating_similarity(ratings: pd.DataFrame, movies: pd.DataFrame):
    """Compute item-item similarity using user rating patterns."""
    pivot = ratings.pivot(index="userId", columns="movieId", values="rating")
    pivot = pivot.reindex(columns=movies["movieId"]).fillna(0.0)

    centered = pivot.subtract(pivot.mean(axis=0), axis=1).fillna(0.0)
    similarity = cosine_similarity(centered.T)
    return np.nan_to_num(similarity)


def build_combined_similarity(genre_sim, rating_sim, genre_weight: float = GENRE_WEIGHT, rating_weight: float = RATING_WEIGHT):
    """Combine genre and rating similarities into a single score matrix."""
    if genre_sim.shape != rating_sim.shape:
        raise ValueError("Genre and rating similarity matrices must have the same shape")
    return genre_sim * genre_weight + rating_sim * rating_weight


movies_df = load_movies()
ratings_df = load_ratings()
movie_stats_df = build_movie_stats(ratings_df, movies_df)
genre_matrix, genre_vectorizer = build_genre_matrix(movies_df)
genre_similarity = build_genre_similarity(genre_matrix)
rating_similarity = build_rating_similarity(ratings_df, movies_df)
combined_similarity = build_combined_similarity(
    genre_similarity,
    rating_similarity
)


def find_movie_index(movie_title: str) -> Optional[int]:
    """Locate a movie by exact title or a case-insensitive substring."""
    title = movie_title.strip().lower()
    exact = movies_df[movies_df["title"].str.lower() == title]
    if not exact.empty:
        return int(exact.index[0])

    fuzzy = movies_df[movies_df["title"].str.contains(movie_title, case=False, regex=False)]
    if not fuzzy.empty:
        return int(fuzzy.index[0])

    return None


def recommend_movie(movie_title: str, top_n: int = 10) -> pd.DataFrame:
    """Return the top N movies that are most relevant to the given title."""
    movie_index = find_movie_index(movie_title)
    if movie_index is None:
        raise ValueError(f"Movie '{movie_title}' not found in dataset.")

    scores = list(enumerate(combined_similarity[movie_index]))
    scores = [
        (idx, score)
        for idx, score in scores
        if idx != movie_index and movie_stats_df.loc[idx, "rating_count"] >= MIN_RATINGS
    ]
    scores = [item for item in scores if item[1] > 0.0]
    scores = sorted(scores, key=lambda item: item[1], reverse=True)

    recommended_indices = [idx for idx, _ in scores[:top_n]]
    recommendations = movies_df.iloc[recommended_indices].copy()
    recommendations["similarity"] = [score for _, score in scores[:top_n]]
    recommendations["rating_count"] = movie_stats_df.loc[recommended_indices, "rating_count"].values
    recommendations["average_rating"] = movie_stats_df.loc[recommended_indices, "average_rating"].values
    return recommendations.reset_index(drop=True)
