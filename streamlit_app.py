# streamlit_app.py
import streamlit as st
import pandas as pd
import gdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

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
            df[c] = ""
    return df


# --- Download and Load the Ratings Data Automatically ---
@st.cache_data(show_spinner=False)
def load_ratings(users_path="steamuser.csv", games_path="gamedata.csv"):
    # Google Drive shareable link for the ratings CSV
    url = "https://drive.usercontent.google.com/download?id=1V_woBuQTiOTxj0OjH0Mx-UhyOY7l14Fg&export=download"
    output = "n_ratings.csv"

    # Download only if not cached locally
    gdown.download(url, output, quiet=False)

    # Load CSVs
    ratings_df = pd.read_csv(output, low_memory=False)
    users_df = pd.read_csv(users_path, low_memory=False)
    games_df = pd.read_csv(games_path, low_memory=False)

    # Sanity check
    if "appid" not in ratings_df.columns or "userID" not in ratings_df.columns:
        st.error("Ratings file must contain 'appid' and 'userID' columns.")
        raise ValueError("Invalid ratings file format.")

    merged_df = pd.merge(ratings_df, games_df[["appid", "name"]], on="appid", how="left")
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
    metadata_df = metadata_df.copy()
    metadata_df["combined_features"] = metadata_df.apply(
        lambda r: combine_features(r, fields_to_include), axis=1
    )

    if description_keywords is None or description_keywords.strip() == "":
        return metadata_df["name"].head(top_n).tolist()

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(metadata_df["combined_features"])
    input_vec = tfidf.transform([description_keywords])
    sim_scores = linear_kernel(input_vec, tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[-top_n:][::-1]
    return metadata_df["name"].iloc[sim_indices].tolist()


def build_user_item_matrix(ratings_df):
    return ratings_df.pivot_table(index="userID", columns="appid", values="rating")


def generate_collaborative_recommendations(selected_game_ratings_list, ratings_df, games_df, top_n=10):
    if not isinstance(selected_game_ratings_list, list) or len(selected_game_ratings_list) == 0:
        return []

    user_item_matrix = build_user_item_matrix(ratings_df)
    if user_item_matrix.shape[0] == 0 or user_item_matrix.shape[1] == 0:
        return []

    item_similarity = cosine_similarity(user_item_matrix.fillna(0).T)
    item_sim_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

    all_game_weighted_scores = {}
    input_appids = []

    for selected, user_rating in selected_game_ratings_list:
        appid = selected if isinstance(selected, (int, str)) else selected
        input_appids.append(appid)

        if appid not in item_sim_df.columns:
            continue

        sim_series = item_sim_df[appid]
        weighted = sim_series * float(user_rating)

        for other_appid, score in weighted.items():
            all_game_weighted_scores.setdefault(other_appid, 0.0)
            all_game_weighted_scores[other_appid] += float(score)

    scores_series = pd.Series(all_game_weighted_scores)
    scores_series = scores_series.drop(index=[a for a in input_appids if a in scores_series.index], errors="ignore")

    if scores_series.empty:
        return []

    top_appids = scores_series.sort_values(ascending=False).head(top_n).index.tolist()
    appid_to_name = games_df.set_index("appid")["name"].to_dict()
    return [appid_to_name.get(a, f"Unknown ({a})") for a in top_appids]


# -------------------------
# App UI
# -------------------------
st.set_page_config(page_title="Game Recommender", layout="wide")
st.title("🎮 Combined Game Recommendation System")

if "mode" not in st.session_state:
    st.session_state.mode = None
if "cf_selected" not in st.session_state:
    st.session_state.cf_selected = []

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Content-Based Filtering"):
        st.session_state.mode = "content"
with col2:
    if st.button("Collaborative Filtering"):
        st.session_state.mode = "collaborative"

games_df = load_games()
try:
    ratings_df, users_df, games_df_ratings, merged_df = load_ratings()
except Exception:
    ratings_df = pd.DataFrame()
    users_df = pd.DataFrame()
    games_df_ratings = games_df.copy()
    merged_df = pd.DataFrame()

# --- Content-Based ---
if st.session_state.mode == "content":
    st.header("Content-Based Filtering")
    fields_to_include = st.multiselect(
        "Select fields:",
        ["description", "genres", "developer", "publisher", "platforms", "required_age", "steamspy_tags"],
        default=["description", "genres", "developer", "publisher", "platforms", "required_age", "steamspy_tags"]
    )
    game_name_filter = st.text_input("Filter by exact game name:")
    description_keywords = st.text_input("Search by description keywords:")
    top_n_cb = st.number_input("Number of recommendations:", 1, 50, 10)

    metadata_filtered = games_df.copy()
    if game_name_filter.strip():
        metadata_filtered = metadata_filtered[metadata_filtered["name"] == game_name_filter.strip()]
        if metadata_filtered.empty:
            st.warning(f"Game name '{game_name_filter}' not found.")

    if st.button("Get Content-Based Recommendations"):
        recs = get_content_based_recommendations_from_metadata(
            metadata_filtered, fields_to_include, description_keywords, top_n=int(top_n_cb)
        )
        if not recs:
            st.write("No recommendations found.")
        else:
            st.write("### Recommendations:")
            for i, r in enumerate(recs, start=1):
                st.write(f"{i}. {r}")

# --- Collaborative ---
if st.session_state.mode == "collaborative":
    st.header("Collaborative Filtering")
    search_query = st.text_input("Search games by name:")
    if st.button("Search Game"):
        matches = games_df[games_df["name"].str.contains(search_query, case=False, na=False)][["appid", "name"]].drop_duplicates()
        if matches.empty:
            st.info("No games matched.")
        else:
            options = [f"{row['name']} (appid: {row['appid']})" for _, row in matches.iterrows()]
            chosen = st.selectbox("Select a game:", options, key="cf_select_search")
            if chosen:
                appid = int(chosen.split("appid:")[1].strip(") "))
                name = chosen.split(" (appid")[0]
                rating = st.slider(f"Rate '{name}' (1-5):", 1, 5, 4, key=f"rating_{appid}")
                if st.button("Add Rated Game"):
                    st.session_state.cf_selected.append((appid, name, rating))
                    st.success(f"Added: {name} with rating {rating}")

    if st.session_state.cf_selected:
        st.subheader("Your Rated Games:")
        df_sel = pd.DataFrame(st.session_state.cf_selected, columns=["appid", "name", "rating"])
        st.dataframe(df_sel.reset_index(drop=True))

        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("Remove Last"):
                if st.session_state.cf_selected:
                    removed = st.session_state.cf_selected.pop()
                    st.info(f"Removed {removed[1]}")
        with colB:
            if st.button("Clear All"):
                st.session_state.cf_selected = []
                st.info("Cleared.")

    top_n_cf = st.number_input("Number of CF recommendations:", 1, 50, 10)
    if st.button("Generate CF Recommendations"):
        if not st.session_state.cf_selected:
            st.warning("Add at least one rated game.")
        else:
            selected_pairs = [(appid, rating) for appid, name, rating in st.session_state.cf_selected]
            recs = generate_collaborative_recommendations(selected_pairs, ratings_df, games_df_ratings, top_n=int(top_n_cf))
            if not recs:
                st.write("No collaborative recommendations could be generated.")
            else:
                st.write("### Collaborative Recommendations:")
                for i, name in enumerate(recs, start=1):
                    st.write(f"{i}. {name}")
