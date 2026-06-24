import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "movies.csv")


def load_movies(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the movie dataset and ensure genre values are strings."""
    movies = pd.read_csv(path)  
    if "title" not in movies.columns or "genres" not in movies.columns:
        raise ValueError("movies.csv must contain 'title' and 'genres' columns")

    movies["genres"] = movies["genres"].fillna("").astype(str)
    return movies


def build_genre_matrix(movies: pd.DataFrame):
    """Convert pipe-separated genres into a machine-readable binary matrix."""
    vectorizer = CountVectorizer(token_pattern=r"[^|]+")
    genre_matrix = vectorizer.fit_transform(movies["genres"])
    return genre_matrix, vectorizer


def build_similarity_matrix(feature_matrix):
    """Compute cosine similarity between every movie pair."""
    return cosine_similarity(feature_matrix, feature_matrix)


movies_df = load_movies()
genre_matrix, genre_vectorizer = build_genre_matrix(movies_df)
similarity_matrix = build_similarity_matrix(genre_matrix)


def find_movie_index(movie_title: str) -> int | None:
    """Find a movie by exact title first, then by case-insensitive substring."""
    lower_title = movie_title.strip().lower()
    exact_matches = movies_df[movies_df["title"].str.lower() == lower_title]
    if not exact_matches.empty:
        return int(exact_matches.index[0])

    partial_matches = movies_df[movies_df["title"].str.contains(movie_title, case=False, regex=False)]
    if not partial_matches.empty:
        return int(partial_matches.index[0])

    return None


def recommend_movie(movie_title: str, top_n: int = 10) -> pd.DataFrame:
    """Return the top N movies most similar to the given title."""
    movie_index = find_movie_index(movie_title)
    if movie_index is None:
        raise ValueError(f"Movie '{movie_title}' not found in dataset.")

    scores = list(enumerate(similarity_matrix[movie_index]))
    scores = sorted(scores, key=lambda item: item[1], reverse=True)

    recommended_indices = [idx for idx, score in scores if idx != movie_index][:top_n]
    recommendations = movies_df.iloc[recommended_indices].copy()
    recommendations["similarity"] = [score for idx, score in scores if idx != movie_index][:top_n]
    return recommendations.reset_index(drop=True)


def format_recommendations(movie_title: str, recommendations: pd.DataFrame) -> str:
    """Format recommendations in a clean, user-friendly string."""
    lines = [f"Top recommendations similar to '{movie_title}':"]
    for _, row in recommendations.iterrows():
        lines.append(
        f"- {row['title']} (genres: {row['genres']}, similarity: {row['similarity']:.2f})"
    )
    return "\n".join(lines)


def main() -> None:
    print("MovieMatch AI - Genre-based Recommendations")
    movie_title = input("Enter a movie title: ").strip()
    if not movie_title:
        print("No movie title entered. Exiting.")
        return

    try:
        recommendations = recommend_movie(movie_title, top_n=10)
        print("\n" + format_recommendations(movie_title, recommendations))
    except ValueError as error:
        print(str(error))
        print("Please try an exact title from the dataset.")


if __name__ == "__main__":
    main()