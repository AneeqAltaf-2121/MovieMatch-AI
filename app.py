from recommender import recommend_movie


def prompt_movie_title() -> str:
    title = input("Enter a movie title to get recommendations: ").strip()
    if not title:
        raise ValueError("Movie title cannot be empty.")
    return title


def display_recommendations(movie_title: str, recommendations):
    if recommendations.empty:
        print(f"No strong recommendations were found for '{movie_title}'.")
        return

    print(f"\nRecommendations for '{movie_title}':")
    for index, row in recommendations.iterrows():
        print(
            f"{index + 1}. {row['title']}\n"
            f"   Genres      : {row['genres']}\n"
            f"   Score       : {row['similarity']:.2f}\n"
            f"   Avg Rating  : {row['average_rating']:.2f}\n"
            f"   Review Count: {int(row['rating_count'])}\n"
        )


def main() -> None:
    print("MovieMatch AI")  
    print("Get smarter movie recommendations based on genres and rating behavior.")

    try:
        movie_title = prompt_movie_title()
        recommendations = recommend_movie(movie_title, top_n=10)
        display_recommendations(movie_title, recommendations)
    except ValueError as error:
        print(f"Error: {error}")
        print("Please enter a valid movie title from the dataset.")


if __name__ == "__main__":
    main()


