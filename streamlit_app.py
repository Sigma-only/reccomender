import streamlit as st
import pandas as pd
import gdown
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# Data Loading
# -------------------------------
@st.cache_data(show_spinner=False)
def load_ratings(users_path="steamuser.csv", games_path="gamedata.csv"):
    file_id = "1V_woBuQTiOTxj0OjH0Mx-UhyOY7l14Fg"
    output = "n_ratings.csv"
    gdown.download(id=file_id, output=output, quiet=False)

    ratings_df = pd.read_csv(output, low_memory=False)
    users_df = pd.read_csv(users_path, low_memory=False)
    games_df = pd.read_csv(games_path, low_memory=False)

    # Normalize columns
    if "userID" not in ratings_df.columns:
        ratings_df.rename(columns={c: "userID" for c in ratings_df.columns if c.lower() in ["user_id", "userid"]}, inplace=True)
    if "appid" not in ratings_df.columns:
        ratings_df.rename(columns={c: "appid" for c in ratings_df.columns if c.lower() in ["app_id", "game_id"]}, inplace=True)

    if "appid" not in ratings_df.columns or "userID" not in ratings_df.columns:
        st.error(f"Ratings file must have user and app id. Found columns: {ratings_df.columns.tolist()}")
        raise ValueError("Invalid ratings file format.")

    merged_df = pd.merge(ratings_df, games_df[["appid", "name"]], on="appid", how="left")
    return ratings_df, users_df, games_df, merged_df


# -------------------------------
# Content-Based Filtering
# -------------------------------
def generate_content_based_recommendations(selected_game, games_df, top_n=5):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(games_df["tags"].fillna(""))

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(games_df.index, index=games_df["name"]).drop_duplicates()

    if selected_game not in indices:
        return []

    idx = indices[selected_game]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: top_n + 1]
    game_indices = [i[0] for i in sim_scores]

    return games_df.iloc[game_indices][["name", "tags"]]


# -------------------------------
# Collaborative Filtering
# -------------------------------
def generate_collaborative_recommendations(user_id, ratings_df, games_df, top_n=5):
    user_game_matrix = ratings_df.pivot_table(index="userID", columns="appid", values="rating").fillna(0)
    if user_id not in user_game_matrix.index:
        return pd.DataFrame(columns=["name", "predicted_score"])

    similarity_matrix = cosine_similarity(user_game_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=user_game_matrix.index, columns=user_game_matrix.index)

    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:6].index
    similar_ratings = user_game_matrix.loc[similar_users]

    mean_ratings = similar_ratings.mean().sort_values(ascending=False)
    recommended_games = mean_ratings.head(top_n).index
    result = games_df[games_df["appid"].isin(recommended_games)][["name"]].copy()
    result["predicted_score"] = [mean_ratings[app] for app in recommended_games]

    return result


# -------------------------------
# Streamlit App UI
# -------------------------------
def main():
    st.title("🎮 Game Recommendation System")

    ratings_df, users_df, games_df, merged_df = load_ratings()

    # Feedback buffer
    if "new_ratings" not in st.session_state:
        st.session_state.new_ratings = []

    option = st.radio("Choose a recommendation method:", ("Content-Based Filtering", "Collaborative Filtering"))

    if option == "Content-Based Filtering":
        st.subheader("Content-Based Game Recommendations")
        game_list = games_df["name"].dropna().unique().tolist()
        selected_game = st.selectbox("Select a game:", game_list)

        if st.button("Get Recommendations"):
            recommendations = generate_content_based_recommendations(selected_game, games_df)
            if not recommendations.empty:
                st.write("Recommended games:")
                st.dataframe(recommendations)
            else:
                st.warning("No recommendations found.")

    elif option == "Collaborative Filtering":
        st.subheader("Collaborative Filtering with Feedback")

        # User ID input
        user_id = st.number_input("Enter your User ID:", min_value=1, step=1)

        # Add rating form
        with st.form("feedback_form"):
            selected_game = st.selectbox("Select a game to rate:", games_df["name"].dropna().unique().tolist())
            rating = st.slider("Rate the game (1–5):", 1, 5, 3)
            submit_rating = st.form_submit_button("Submit Rating")

        if submit_rating:
            appid = games_df.loc[games_df["name"] == selected_game, "appid"].values[0]
            new_row = {"userID": user_id, "appid": appid, "rating": rating}
            st.session_state.new_ratings.append(new_row)
            st.success(f"Added rating: {selected_game} ({rating}/5)")

        # Combine original + new ratings
        ratings_with_feedback = pd.concat(
            [ratings_df, pd.DataFrame(st.session_state.new_ratings)],
            ignore_index=True
        )

        if st.button("Get Recommendations"):
            recommendations = generate_collaborative_recommendations(user_id, ratings_with_feedback, games_df)
            if not recommendations.empty:
                st.write("Recommended games:")
                st.dataframe(recommendations)
            else:
                st.warning("No recommendations found for this user.")


if __name__ == "__main__":
    main()
