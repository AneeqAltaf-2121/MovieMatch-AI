# MovieMatch AI

MovieMatch AI is a movie recommendation system that suggests similar movies based on genre similarity and user rating behavior.

## Features

- Recommends movies from a user-entered title
- Combines genre-based similarity with rating-based similarity
- Filters out movies with low review counts
- Displays similarity score, average rating, and review count

## Dataset

This project uses the MovieLens small dataset.

Files used:

- movies.csv
- ratings.csv

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn

## How It Works

1. Loads movie and rating data
2. Converts genres into machine-readable vectors
3. Computes genre similarity
4. Computes rating-based item-item similarity
5. Combines both similarity scores
6. Returns the top recommended movies

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt