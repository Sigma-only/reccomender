import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity





# --- Content-Based Filtering Section ---
metadata = pd.read_csv('gamedata.csv', low_memory=False)

# Fill NaN values for necessary columns
for col in ['short_description', 'developer', 'publisher', 'platforms', 'required_age', 'categories', 'genres', 'steamspy_tags', 'detailed_description', 'about_the_game']:
    metadata[col] = metadata[col].fillna('')

# Content-Based Filtering Functions
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


# Streamlit UI
st.title('Game Recommendation System')

# User inputs
game_name = st.text_input("Enter Game Name (Optional)")
description_keywords = st.text_input("Enter Keywords for Game Description (Optional)")
developer = st.text_input("Enter Game Developer (Optional)")
publisher = st.text_input("Enter Game Publisher (Optional)")
platforms = st.text_input("Enter Game Platforms (Optional)")
required_age = st.number_input("Enter Required Age (Optional)", min_value=0, max_value=18, step=1)
genres = st.text_input("Enter Game Genres (Optional)")
steamspy_tags_input = st.text_input("Enter SteamSpy Tags (Optional)")

# Generate recommendations button
if st.button("Generate Recommendations"):
    recommendations = get_content_based_recommendations(
        game_name=game_name,
        description_keywords=description_keywords,
        developer=developer,
        publisher=publisher,
        platforms=platforms,
        required_age=required_age,
        genres=genres,
        steamspy_tags_input=steamspy_tags_input
    )

    if isinstance(recommendations, str):
        st.error(recommendations)
    else:
        st.write("Recommended Games:")
        st.write(recommendations)

