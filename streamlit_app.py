# streamlit_app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import gdown

# --- Download and Load the Ratings Data Automatically ---
@st.cache_data
def load_ratings():
    # Google Drive shareable link for the ratings CSV
    url = "https://drive.usercontent.google.com/download?id=1V_woBuQTiOTxj0OjH0Mx-UhyOY7l14Fg&export=download"  # Update with your correct file ID
    output = "n_rating.csv"
    
    # Download the CSV file from Google Drive
    gdown.download(url, output, quiet=False)
    
    # Load ratings CSV into pandas DataFrame
    ratings_path = pd.read_csv(output)
    return ratings_df


# -------------------------
# Data loading (cached)
# -------------------------
@st.cache_data(show_spinner=False)
def load_games(path="gamedata.csv"):
    try:
        df = pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        st.error(f"Could not find `{path}`. Place it in the app folder or repository root.")
        raise
    # ensure expected text columns exist (fillna)
    text_cols = [
        "short_description", "developer", "publisher", "platforms",
        "required_age", "categories", "genres", "steamspy_tags",
        "detailed_description", "about_the_game", "name", "appid"
    ]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].fillna("")
        else:
            # keep missing columns but create empty to avoid key errors later
            df[c] = ""
    return df

@st.cache_data(show_spinner=False)
def load_ratings(ratings_path="n_ratings.csv", users_path="steamuser.csv", games_path="gamedata.csv"):
    try:
        ratings_df = pd.read_csv(ratings_path, low_memory=False)
        users_df = pd.read_csv(users_path, low_memory=False)
        games_df = pd.read_csv(games_path, low_memory=False)
    except FileNotFoundError as e:
        st.error(f"Missing data file: {e}")
        raise

    # Try to merge like in the notebook (on appid and userID)
    merged_df = ratings_df.copy()
    # If ratings have appid and userID columns, keep them; otherwise try to infer
    if "appid" not in merged_df.columns or "userID" not in merged_df.columns:
        st.error("Ratings file must contain 'appid' and 'userID' columns.")
        raise ValueError("Invalid ratings file format.")
    # basic merge with games and users so we can show names, etc.
    merged_df = pd.merge(merged_df, games_df[["appid", "name"]], on="appid", how="left")
    # optionally merge users if needed (not required for computation)
    return ratings_df, users_df, games_df, merged_df

# -------------------------
# Helper functions
# -------------------------
def combine_features(row, fields_to_include):
    features = []
    if "description" in fields_to_include:
        features.append(row.get("short_description", ""))
        features.append(row.get("detailed_description", ""))
        features.append(row.get("about_the_game", ""))
    if "genres" in fields_to_include:
        features.append(row.get("genres", ""))
    if "developer" in fields_to_include:
        features.append(row.get("developer", ""))
    if "publisher" in fields_to_include:
        features.append(row.get("publisher", ""))
    if "platforms" in fields_to_include:
        features.append(row.get("platforms", ""))
    if "required_age" in fields_to_include:
        features.append(str(row.get("required_age", "")))
    if "steamspy_tags" in fields_to_include:
        features.append(row.get("steamspy_tags", ""))
    return " ".join([str(f) for f in features if f is not None])

def get_content_based_recommendations_from_metadata(
    metadata_df, fields_to_include, description_keywords="", top_n=10
):
    # build combined features (keeps attributes)
    metadata_df = metadata_df.copy()
    metadata_df["combined_features"] = metadata_df.apply(lambda r: combine_features(r, fields_to_include), axis=1)

    if description_keywords is None or description_keywords.strip() == "":
        # return top names if no keywords (matches notebook behavior: simply return filtered list)
        return metadata_df["name"].head(top_n).tolist()

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(metadata_df["combined_features"])
    input_vec = tfidf.transform([description_keywords])
    sim_scores = linear_kernel(input_vec, tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[-top_n:][::-1]
    return metadata_df["name"].iloc[sim_indices].tolist()

def build_user_item_matrix(ratings_df):
    # pivot: rows = userID, columns = appid, values = rating
    user_item = ratings_df.pivot_table(index="userID", columns="appid", values="rating")
    return user_item

def generate_collaborative_recommendations(selected_game_ratings_list, ratings_df, games_df, top_n=10):
    """
    selected_game_ratings_list: list of tuples (appid, rating) OR (game_row_df, rating).
       We'll normalize by accepting either.
    ratings_df: original ratings dataframe (with columns userID, appid, rating)
    games_df: dataframe with appid->name mapping
    Returns: list of recommended game names (top_n), in descending order of score.
    """
    if not isinstance(selected_game_ratings_list, list) or len(selected_game_ratings_list) == 0:
        return []

    # Build user-item matrix
    user_item_matrix = build_user_item_matrix(ratings_df)
    # If all NaN or empty matrix, return empty
    if user_item_matrix.shape[0] == 0 or user_item_matrix.shape[1] == 0:
        return []

    # Compute item-item similarity matrix (cosine) using columns (items) vectors
    # Replace NaN with 0 for similarity computation
    item_similarity = cosine_similarity(user_item_matrix.fillna(0).T)
    item_sim_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

    all_game_weighted_scores = {}

    # For each selected game & user rating, accumulate weighted similarity scores
    input_appids = []
    for selected, user_rating in selected_game_ratings_list:
        # accept either an appid integer/string or a DataFrame/Series with 'appid'
        if isinstance(selected, (int, str)):
            appid = selected
        elif isinstance(selected, dict):
            # if user passed dict like {'appid': 123}
            appid = selected.get("appid")
        else:
            # assume a dataframe or series with appid column or index 0
            try:
                appid = selected["appid"] if "appid" in selected else selected.iloc[0]["appid"]
            except Exception:
                # try to convert to int
                appid = selected

        input_appids.append(appid)
        # if appid not in item_sim_df columns, skip
        if appid not in item_sim_df.columns:
            # skip unknown items (maybe new appid not in ratings matrix)
            continue

        sim_series = item_sim_df[appid]  # similarity of this selected item to every other item
        # Use the raw rating as weight (same behavior as notebook snippet)
        weight = float(user_rating)
        weighted = sim_series * weight

        # accumulate
        for other_appid, score in weighted.items():
            # initialize
            all_game_weighted_scores.setdefault(other_appid, 0.0)
            all_game_weighted_scores[other_appid] += float(score)

    # Convert to series
    scores_series = pd.Series(all_game_weighted_scores)
    # Drop items user already rated (input_appids)
    scores_series = scores_series.drop(index=[a for a in input_appids if a in scores_series.index], errors="ignore")

    if scores_series.empty:
        return []

    # Sort descending and return top_n mapped to names
    top_appids = scores_series.sort_values(ascending=False).head(top_n).index.tolist()
    appid_to_name = games_df.set_index("appid")["name"].to_dict()
    recommendations = [appid_to_name.get(a, f"Unknown ({a})") for a in top_appids]
    return recommendations

# -------------------------
# App UI: Buttons like notebook
# -------------------------
st.set_page_config(page_title="Game Recommender", layout="wide")
st.title("🎮 Combined Game Recommendation System")

# Use session_state to persist which mode is chosen and CF selection list
if "mode" not in st.session_state:
    st.session_state.mode = None
if "cf_selected" not in st.session_state:
    st.session_state.cf_selected = []  # list of tuples (appid, name, rating)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Content-Based Filtering"):
        st.session_state.mode = "content"
with col2:
    if st.button("Collaborative Filtering"):
        st.session_state.mode = "collaborative"

# load datasets (on-demand)
games_df = load_games()
try:
    ratings_df, users_df, games_df_ratings, merged_df = load_ratings()
except Exception:
    # load_ratings already reported the error to user; stop further UI for CF if files missing
    ratings_df = pd.DataFrame()
    users_df = pd.DataFrame()
    games_df_ratings = games_df.copy()
    merged_df = pd.DataFrame()

# -------------------------
# Content-Based Section (renders if mode == 'content')
# -------------------------
if st.session_state.mode == "content":
    st.header("Content-Based Filtering")
    st.write("This section uses TF-IDF over combined features to recommend games.")

    fields_to_include = st.multiselect(
        "Select fields to include in combined features:",
        ["description", "genres", "developer", "publisher", "platforms", "required_age", "steamspy_tags"],
        default=["description", "genres", "developer", "publisher", "platforms", "required_age", "steamspy_tags"]
    )
    game_name_filter = st.text_input("Filter by exact game name (leave blank to skip):")
    description_keywords = st.text_input("Search by description keywords (optional):")
    top_n_cb = st.number_input("Number of recommendations to show:", min_value=1, max_value=50, value=10)

    # Apply name filter if provided (mimics notebook filtering)
    metadata_filtered = games_df.copy()
    if game_name_filter.strip():
        metadata_filtered = metadata_filtered[metadata_filtered["name"] == game_name_filter.strip()]
        if metadata_filtered.empty:
            st.warning(f"Game name '{game_name_filter}' not found.")
    # run recommendation (even if filtered down to single game)
    if st.button("Get Content-Based Recommendations"):
        recs = get_content_based_recommendations_from_metadata(metadata_filtered, fields_to_include, description_keywords, top_n=int(top_n_cb))
        if not recs:
            st.write("No recommendations found with given filters / keywords.")
        else:
            st.write("### Recommendations:")
            for i, r in enumerate(recs, start=1):
                st.write(f"{i}. {r}")

# -------------------------
# Collaborative Section (renders if mode == 'collaborative')
# -------------------------
if st.session_state.mode == "collaborative":
    st.header("Collaborative Filtering (rate games, then generate CF)")
    st.write("Search for games, add them to your 'rated' list with a rating, then click Generate.")

    # Search box
    search_query = st.text_input("Search games by name (substring):", key="cf_search")
    if st.button("Search Game"):
        if not search_query.strip():
            st.warning("Type a search term first.")
        else:
            matches = games_df[games_df["name"].str.contains(search_query, case=False, na=False)][["appid", "name"]].drop_duplicates()
            if matches.empty:
                st.info("No games matched your search.")
            else:
                # let user pick one of the matches
                options = [f"{row['name']}  (appid: {row['appid']})" for _, row in matches.iterrows()]
                chosen = st.selectbox("Select a game from search results:", options, key="cf_select_search")
                # parse chosen to appid
                if chosen:
                    # display the appid and name and allow rating
                    import re
                    m = re.search(r"\(appid:\s*([0-9]+)\)", chosen)
                    if m:
                        chosen_appid = int(m.group(1))
                    else:
                        # fallback: try to match by name
                        chosen_row = matches[matches["name"] == chosen.split("  (appid")[0]]
                        chosen_appid = int(chosen_row["appid"].iloc[0]) if not chosen_row.empty else None

                    chosen_name = chosen.split("  (appid")[0]
                    if chosen_appid is None:
                        st.error("Could not determine appid for the selected game.")
                    else:
                        rating = st.slider(f"Rate '{chosen_name}' (1-5):", min_value=1, max_value=5, value=4, key=f"rating_{chosen_appid}")
                        if st.button("Add Rated Game"):
                            # append to session list
                            st.session_state.cf_selected.append((chosen_appid, chosen_name, int(rating)))
                            st.success(f"Added: {chosen_name} (appid {chosen_appid}) with rating {rating}")

    # Show current selected list
    if st.session_state.cf_selected:
        st.subheader("Currently rated games (your input):")
        df_sel = pd.DataFrame(st.session_state.cf_selected, columns=["appid", "name", "rating"])
        st.dataframe(df_sel.reset_index(drop=True))
        # option to remove last or clear
        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("Remove last rated game"):
                if st.session_state.cf_selected:
                    removed = st.session_state.cf_selected.pop()
                    st.info(f"Removed {removed[1]}")
        with colB:
            if st.button("Clear all rated games"):
                st.session_state.cf_selected = []
                st.info("Cleared all rated games.")

    st.write("---")
    # Generate recommendations
    top_n_cf = st.number_input("Number of CF recommendations:", min_value=1, max_value=50, value=10)

    if st.button("Generate CF Recommendations"):
        if not st.session_state.cf_selected:
            st.warning("Add at least one rated game before generating collaborative recommendations.")
        else:
            # prepare input list as (appid, rating)
            selected_pairs = [(int(appid), int(rating)) for appid, name, rating in st.session_state.cf_selected]
            try:
                recs = generate_collaborative_recommendations(selected_pairs, ratings_df, games_df_ratings, top_n=int(top_n_cf))
            except Exception as e:
                st.error(f"Error while generating collaborative recommendations: {e}")
                recs = []

            if not recs:
                st.write("No collaborative recommendations could be generated (insufficient data or no similar items).")
            else:
                st.write("### Collaborative Recommendations:")
                for i, name in enumerate(recs, start=1):
                    st.write(f"{i}. {name}")

# Footer / help
st.write("---")
st.markdown(
    "Notes:\n"
    "- Make sure `gamedata.csv`, `n_ratings.csv`, and `steamuser.csv` are available in the app repository.\n"
    "- Collaborative filtering follows the notebook approach: it computes item-item (game-game) similarity from the existing ratings matrix and weights similar items by your provided ratings.\n"
)
