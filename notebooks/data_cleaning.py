import pandas as pd

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

print("Movies Shape:", movies.shape)
print("Ratings Shape:", ratings.shape)

print("\nMovies Missing Values:")
print(movies.isnull().sum())

print("\nRatings Missing Values:")
print(ratings.isnull().sum())

print("\nDuplicate Movies:", movies.duplicated().sum())
print("Duplicate Ratings:", ratings.duplicated().sum())

movie_ratings = movies.merge(
    ratings,
    on="movieId"
)

print("\nMerged Dataset Shape:")
print(movie_ratings.shape)

print("\nMerged Dataset Preview:")
print(movie_ratings.head())