# MovieMatch AI

MovieMatch AI is a movie recommendation system built with Python, Pandas, NumPy, and Scikit-Learn. The project recommends similar movies by combining genre-based similarity with user rating behavior from the MovieLens dataset.

## Features

* Recommends movies from a user-entered title
* Combines genre-based similarity and rating-based similarity
* Filters out movies with low review counts
* Displays recommendation score, average rating, and review count
* Command-line interface for easy interaction

## Dataset

This project uses the MovieLens Small Dataset.

Files used:

* movies.csv
* ratings.csv

Dataset Source:
https://grouplens.org/datasets/movielens/

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Git
* GitHub

## Project Structure

```text
MovieMatch-AI
│
├── data
│   ├── movies.csv
│   └── ratings.csv
│
├── notebooks
│   ├── data_exploration.py
│   └── data_cleaning.py
│
├── app.py
├── recommender.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How It Works

1. Loads movie and rating data
2. Converts movie genres into machine-readable vectors
3. Computes genre similarity using cosine similarity
4. Computes rating-based item-item similarity
5. Combines both similarity scores
6. Returns the most relevant movie recommendations

## Installation

Clone the repository:

```bash
git clone https://github.com/AneeqAltaf-2121/MovieMatch-AI.git
cd MovieMatch-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

## Example Output

Input:

```text
Toy Story
```

Output:

```text
1. Toy Story 2 (1999)
2. Monsters, Inc. (2001)
3. Bug's Life, A (1998)
4. Shrek (2001)
5. Toy Story 3 (2010)
```

## Skills Demonstrated

* Data Cleaning
* Data Exploration
* Recommendation Systems
* Cosine Similarity
* Feature Engineering
* Pandas Data Processing
* Python Development
* Git Version Control
* GitHub Workflow

## Future Improvements

* Streamlit Web Interface
* FastAPI Backend
* User Profiles
* Personalized Recommendations
* Movie Poster Integration
* Deployment to the Cloud

## Author

Aneeq Altaf

GitHub:
https://github.com/AneeqAltaf-2121

## Project Status

Completed 
