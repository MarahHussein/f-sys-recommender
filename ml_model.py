import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()  # This loads environment variables from .env file
import numpy as np

def clean_dataframe(df):
    """Replace all 'inf', '-inf', and 'nan' with None (which becomes 'null' in JSON)."""
    return df.replace([np.inf, -np.inf, np.nan], None)

# Establish connection to MongoDB
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
database = client.Movie_Recommendation
collection = database.CSVData

# Fetch data and convert to DataFrame
data = list(collection.find({}))
movies = pd.DataFrame(data)
movies.drop(columns=['_id'], inplace=True)  # Optional: remove MongoDB's auto-generated ID

# Close the MongoDB connection after fetching data
client.close()

# Ensure non-null values for text processing
movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].fillna('')
movies['keywords'] = movies['keywords'].fillna('')
movies['combined_features'] = movies['overview'] + ' ' + movies['genres'] + ' ' + movies['keywords']

# Vectorize the combined text features
vectorizer = TfidfVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(movies['combined_features'])

# Compute cosine similarity between all movie pairs
movie_similarity = cosine_similarity(feature_matrix)

# Convert to DataFrame for easier manipulation
movie_similarity_df = pd.DataFrame(movie_similarity, index=movies['id'].astype(str), columns=movies['id'].astype(str))


def recommend_movies(movie_ids, num_recommendations=3):
    notFind = []
    recommendations = []
    for movie_id in movie_ids:
        movie_id = str(movie_id)
        if movie_id not in movie_similarity_df.index:
            notFind.append(movie_id)
            continue

        if movie_id in movie_similarity_df.index:
            # Get movies sorted by similarity scores
            similar_movies = movie_similarity_df.loc[movie_id].sort_values(ascending=False)
    
    # Exclude the movie itself and get the top N recommendations
            top_recommendations = similar_movies.iloc[1:num_recommendations+1]

    # Map indices to movie titles
            recommended_movie_ids = top_recommendations.index
            recommended_movies = movies[movies['id'].astype(str).isin(recommended_movie_ids)]
            r = clean_dataframe(recommended_movies)
            recommended_movies_list = r.to_dict(orient='records')
            recommendations.extend(recommended_movies_list)

    return {"not_found": notFind, "recommended": recommendations}
