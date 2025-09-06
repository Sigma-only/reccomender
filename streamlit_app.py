import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# --- Content-Based Filtering Section ---
@st.cache_data
def load_data():
    metadata = pd.read_csv('gamedata.csv', low_memory=False)

    # Fill NaN values for necessary columns
    for col in ['short_description', 'developer', 'publisher', 'platforms', 'required_age', 'categories', 'genres', 'steamspy_tags', 'detailed_description', 'about_the_game']:
        metadata[col] = metadata[col].fillna('')

    return metadata

metadata = load_data()

def combine_features(row, fields_to_include):
    features = []

    if 'description' in fields_to_include:
        features.append(row['short_description'])
        features.append(row['detailed_description'])
        features.append(row['about_the_game'])
    if 'genres' in fields_to_include:
        features.append(row['genres'])
    if 'developer' in fields_to_include:
        features.append(row['developer'])
    if 'publisher' in fields_to_include:
        features.append(row['publisher'])
    if 'platforms' in fields_to_include:
        features.append(row['platforms'])
    if 'required_age' in fields_to_include:
        features.append(str(row['required_age']))
    if 'steamspy_tags' in fields_to_include:
        features.append(row['steamspy_tags'])

    return ' '.join(features)

# Fields user can include for the recommendation
fields_to_include = ['description', 'genres', 'developer', 'publisher', 'platforms', 'required_age', 'steamspy_tags']

# Apply the combine_features function based on the selected fields
metadata['combined_features'] = metadata.apply(lambda row: combine_features(row, fields_to_include), axis=1)

# Vectorize the combined features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(metadata['combined_features'])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create indices for quick lookup
indices = pd.Series(metadata.index, index=metadata['name']).drop_duplicates()

def get_content_based_recommendations(game_name='', description_keywords='', developer='', publisher='', platforms='', required_age=None, genres='', steamspy_tags_input='', cosine_sim=cosine_sim, metadata=metadata, top_n=50):

    filtered_metadata = metadata.copy()

    if game_name.strip():
        filtered_metadata = filtered_metadata[filtered_metadata['name'] == game_name]
        if filtered_metadata.empty:
            return f"Game name '{game_name}' not found in dataset."

    if developer.strip():
        filtered_metadata = filtered_metadata[filtered_metadata['developer'].str.contains(developer, case=False, na=False)]

    if publisher.strip():
        filtered_metadata = filtered_metadata[filtered_metadata['publisher'].str.contains(publisher, case=False, na=False)]

    if platforms.strip():
        filtered_metadata = filtered_metadata[filtered_metadata['platforms'].str.contains(platforms, case=False, na=False)]

    if required_age is not None:
        filtered_metadata = filtered_metadata[filtered_metadata['required_age'] == required_age]

    elif genres.strip():
         filtered_metadata = filtered_metadata[filtered_metadata['genres'].str.contains(genres, case=False, na=False)]

    if steamspy_tags_input.strip():
        tags = steamspy_tags_input.lower().split()
        steamspy_tags_logic = 'and'
        if 'or' in tags:
            steamspy_tags_logic = 'or'
            tags.remove('or')
        elif 'and' in tags:
            tags.remove('and')

        if not tags:
            return "No valid SteamSpy tags found in your input."

        if steamspy_tags_logic == 'and':
            for tag in tags:
                filtered_metadata = filtered_metadata[filtered_metadata['steamspy_tags'].str.contains(tag, case=False, na=False)]
        elif steamspy_tags_logic == 'or':
            or_filter = filtered_metadata['steamspy_tags'].str.contains(tags[0], case=False, na=False)
            for tag in tags[1:]:
                or_filter = or_filter | filtered_metadata['steamspy_tags'].str.contains(tag, case=False, na=False)
            filtered_metadata = filtered_metadata[or_filter]

    if not description_keywords.strip():
        if filtered_metadata.empty:
            return "No games found matching the specified filters."
        else:
            return filtered_metadata['name'].head(top_n)

    if filtered_metadata.empty:
        return "No games found matching the specified filters."

    tfidf_filtered = TfidfVectorizer(stop_words='english')
    tfidf_matrix_filtered = tfidf_filtered.fit_transform(filtered_metadata['combined_features'])

    input_vec = tfidf_filtered.transform([description_keywords])

    sim_scores = linear_kernel(input_vec, tfidf_matrix_filtered).flatten()

    sim_indices = sim_scores.argsort()[-top_n:][::-1]

    return filtered_metadata['name'].iloc[sim_indices]


# --- Collaborative Filtering Section ---
@st.cache_data
def load_rating_data():
    ratings_df = pd.read_csv('n_ratings.csv')
    return ratings_df

ratings_df = load_rating_data()

# Create the user-item interaction matrix for collaborative filtering
def create_user_item_matrix(ratings_df):
    user_item_matrix = ratings_df.pivot_table(index='user_id', columns='game_id', values='rating')
    return user_item_matrix.fillna(0)

user_item_matrix = create_user_item_matrix(ratings_df)

# Generate collaborative recommendations based on user ratings
def generate_collaborative_recommendations(user_ratings, user_item_matrix, k=10):
    user_item_matrix = csr_matrix(user_item_matrix.values)
    
    knn = NearestNeighbors(metric='cosine', n_neighbors=k)
    knn.fit(user_item_matrix)
    
    user_vector = np.array(user_ratings).reshape(1, -1)
    distances, indices = knn.kneighbors(user_vector, n_neighbors=k)
    
    recommended_games = []
    for i in range(k):
        game_id = user_item_matrix.columns[indices[0][i]]
        recommended_games.append(game_id)
    
    return recommended_games


# --- Streamlit UI ---
def run_recommendation_system():
    st.title("Game Recommendation System")

    st.sidebar.header("Content-Based Filtering")
    fields_to_include_selected = st.sidebar.multiselect(
        "Select criteria to include in recommendations:",
        options=['description', 'genres', 'developer', 'publisher', 'platforms', 'required_age', 'steamspy_tags'],
        default=['description', 'genres', 'developer', 'publisher']
    )

    game_name = st.text_input("Game Name:")
    description_keywords = st.text_input("Description Keywords:")
    developer = st.text_input("Developer:")
    publisher = st.text_input("Publisher:")
    platforms = st.text_input("Platforms:")
    required_age = st.slider("Required Age:", 0, 18, 0)
    steamspy_tags_input = st.text_input("SteamSpy Tags (use space-separated values):")

    # When the user clicks on the "Generate Recommendations" button
    if st.button('Generate Content-Based Recommendations'):
        st.write("Generating content-based recommendations...")

        user_inputs_cb = {
            'game_name': game_name,
            'description_keywords': description_keywords,
            'developer': developer,
            'publisher': publisher,
            'platforms': platforms,
            'required_age': required_age,
            'steamspy_tags_input': steamspy_tags_input
        }

        recommended_games_cb = get_content_based_recommendations(**user_inputs_cb, top_n=50)

        if isinstance(recommended_games_cb, str):
            st.error(recommended_games_cb)
        else:
            for idx, game in enumerate(recommended_games_cb, 1):
                st.write(f"{idx}. {game}")

    # Collaborative Filtering Section
    st.sidebar.header("Collaborative Filtering")
    user_ratings = []
    games_to_rate = metadata['name'].head(10)  # Show top 10 games to rate

    for game in games_to_rate:
        rating = st.slider(f"Rate the game '{game}'", 0, 5, 3)
        user_ratings.append(rating)

    if st.button('Generate Collaborative Recommendations'):
        st.write("Generating collaborative recommendations...")

        recommended_games_cf = generate_collaborative_recommendations(user_ratings, user_item_matrix, k=10)

        st.write("Recommended Games based on your ratings:")

        for game_id in recommended_games_cf:
            game_name = metadata[metadata['appid'] == game_id]['name'].values[0]
            st.write(f"- {game_name}")


if __name__ == "__main__":
    run_recommendation_system()
