import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import csr_matrix

# --- Load Data ---
@st.cache_data
def load_data():
    metadata = pd.read_csv("gamedata.csv", low_memory=False)
    ratings_df = pd.read_csv("n_ratings.csv")
    users_df = pd.read_csv("steamuser.csv")

    # Fill missing values
    for col in ['short_description', 'developer', 'publisher', 'platforms', 'required_age', 'categories', 'genres', 'steamspy_tags', 'detailed_description', 'about_the_game']:
        metadata[col] = metadata[col].fillna('')

    return metadata, ratings_df, users_df

metadata, ratings_df, users_df = load_data()
games_df = metadata.copy()  # Keep games_df for CF

# --- Prepare Content-Based ---
def combine_features(row, fields):
    features = []
    if 'description' in fields:
        features.extend([row['short_description'], row['detailed_description'], row['about_the_game']])
    if 'genres' in fields: features.append(row['genres'])
    if 'developer' in fields: features.append(row['developer'])
    if 'publisher' in fields: features.append(row['publisher'])
    if 'platforms' in fields: features.append(row['platforms'])
    if 'required_age' in fields: features.append(str(row['required_age']))
    if 'steamspy_tags' in fields: features.append(row['steamspy_tags'])
    return ' '.join(features)

fields_to_include = ['description', 'genres', 'developer', 'publisher', 'platforms', 'required_age', 'steamspy_tags']
metadata['combined_features'] = metadata.apply(lambda row: combine_features(row, fields_to_include), axis=1)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(metadata['combined_features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(metadata.index, index=metadata['name']).drop_duplicates()

# --- Content-Based Function ---
def get_content_based_recommendations(description_keywords='', top_n=10):
    if not description_keywords.strip():
        return "Please enter keywords."

    tfidf_input = tfidf.transform([description_keywords])
    sim_scores = linear_kernel(tfidf_input, tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[-top_n:][::-1]
    return metadata['name'].iloc[sim_indices]

# --- Collaborative Filtering Prep ---
merged_df = pd.merge(ratings_df, metadata, on='appid')
merged_df = pd.merge(merged_df, users_df, on='userID')

user_item_matrix = merged_df.pivot_table(index='userID', columns='appid', values='rating').fillna(0)
user_item_sparse_matrix = csr_matrix(user_item_matrix.values)

# --- Collaborative Filtering Function ---
def generate_collaborative_recommendations(game_name, user_rating, simulate_max_score=False):
    game_name_lower = game_name.lower()
    found_games = games_df[games_df['name'].str.lower().str.contains(game_name_lower, na=False)]

    if found_games.empty:
        return "Game not found."
    
    selected_game = found_games.iloc[0]
    appid = selected_game['appid']
    
    item_similarity = cosine_similarity(user_item_matrix.T)
    similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    
    try:
        game_scores = similarity_df[appid] * user_rating
        game_scores = game_scores.sort_values(ascending=False)
        game_scores = game_scores.drop(appid)
        top_games = game_scores.head(10).index
        recommendations = games_df[games_df['appid'].isin(top_games)]
        return recommendations[['name']]
    except KeyError:
        return "Game not found in rating matrix."

# --- Streamlit UI ---
st.title("🎮 Game Recommendation System")

mode = st.radio("Select Recommendation Type", ['Content-Based', 'Collaborative Filtering'])

if mode == 'Content-Based':
    st.subheader("🔍 Content-Based Filtering")
    keywords = st.text_input("Enter game description or tags:")
    if st.button("Get Recommendations"):
        results = get_content_based_recommendations(description_keywords=keywords, top_n=10)
        st.write("### Recommended Games:")
        st.write(results)

elif mode == 'Collaborative Filtering':
    st.subheader("🤝 Collaborative Filtering")
    game_name = st.text_input("Enter a game you've played:")
    rating = st.slider("Your rating for the game (1-5):", 1, 5, 3)
    if st.button("Get Recommendations"):
        results = generate_collaborative_recommendations(game_name, rating)
        st.write("### Recommended Games:")
        st.write(results)
