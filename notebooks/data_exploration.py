import pandas as pd

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

print("Movies Dataset:")
print(movies.head())
print()

print("Ratings Dataset:")
print(ratings.head())
print()

print("Movies shape:", movies.shape)
print("Ratings shape:", ratings.shape)
print()

print("Movies columns:")
print(movies.columns)
print()

print("Ratings columns:")
print(ratings.columns)
print()

print("Missing values in movies:")
print(movies.isnull().sum())
print()

print("Missing values in ratings:")
print(ratings.isnull().sum())