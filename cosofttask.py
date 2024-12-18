import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
# Sample movie dataset (movies and ratings)
movies = pd.DataFrame({
    'movieId': [1, 2, 3, 4, 5, 6],
    'title': ['The Matrix', 'Inception', 'The Godfather', 'Toy Story', 'Avengers', 'The Dark Knight'],
    'genre': ['Action', 'Sci-Fi', 'Crime', 'Animation', 'Action', 'Action']
})

# Sample user ratings (users and movie ratings)
ratings = pd.DataFrame({
    'userId': [1, 1, 2, 2, 3, 3, 4, 4, 5],
    'movieId': [1, 2, 1, 3, 2, 4, 3, 5, 5],
    'rating': [5, 4, 4, 5, 3, 5, 4, 2, 4]
})
# Create a pivot table (user-item matrix)
user_movie_ratings = ratings.pivot_table(index='userId', columns='movieId', values='rating')

# Fill NaN with 0 (assumption: missing ratings means the user has not watched the movie)
user_movie_ratings = user_movie_ratings.fillna(0)

# Calculate the cosine similarity between movies
cosine_sim = cosine_similarity(user_movie_ratings.T)

# Convert cosine similarity into a DataFrame for easy readability
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_movie_ratings.columns, columns=user_movie_ratings.columns)

def recommend_movies_based_on_item(movie_id, top_n=3):
    # Get the similarity scores for the given movie
    similar_movies = cosine_sim_df[movie_id]
    
    # Sort the movies based on similarity score
    similar_movies = similar_movies.sort_values(ascending=False)
    
    # Get the top_n most similar movies
    recommended_movie_ids = similar_movies.index[1:top_n+1]  # Exclude the movie itself (index 0)
    
    # Retrieve the movie titles
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]['title']
    return recommended_movies

# Example: Recommend movies similar to 'The Matrix' (movieId = 1)
recommend_movies_based_on_item(1)

