import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib
import re

DATA_FILE = 'spotify_song_audio_features_norm_25k.csv'

AUDIO_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo', 'duration_ms'
]
SENTIMENT_FEATURE = ['sentiment']

def filter_similar_variants(results_df):
    """
    Filters out duplicate versions (Radio Edits, Remasters, Features)
    while preserving covers (same title, different primary artist).
    """
    df = results_df.copy()

    # gets the main part of the title (excluding (Remaster version), (Radio edit), (feat. X), etc)
    def clean_title(text):
        text = str(text).lower()
        # Remove text in parentheses/brackets and specific suffixes
        text = re.sub(r"\(.*?\)", "", text) 
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\-.*remaster.*", "", text)
        text = re.sub(r"\-.*radio edit.*", "", text)
        return text.strip()

    # gets "artist" - "featured artist"
    def get_primary_artist(text):
        text = str(text)
        # Clean list formatting and take the first artist
        text = text.replace("['", "").replace("']", "").replace("', '", ",")
        return text.split(",")[0].strip().lower()

    df['clean_title_temp'] = df['name'].apply(clean_title)
    df['primary_artist_temp'] = df['artists'].apply(get_primary_artist)

    # Drop duplicate if both Clean Title AND Primary Artist match
    df_filtered = df.drop_duplicates(subset=['clean_title_temp', 'primary_artist_temp'], keep='first')

    return df_filtered.drop(columns=['clean_title_temp', 'primary_artist_temp'])

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_FILE)
        df['search_label'] = df['artists'] + " - " + df['name']
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error(f"ERROR: Could not find dataset: '{DATA_FILE}'.")
    st.stop()


st.sidebar.title("Find Similar Music!")

# Choosing initial songs
st.sidebar.header("Build Seed Playlist")
if 'selected_songs' not in st.session_state:
    st.session_state['selected_songs'] = []

search_query = st.sidebar.text_input("Filter by Artist or Title:", placeholder="Search...")

hits = []
if search_query:
    hits = df[df['search_label'].str.contains(search_query, case=False, na=False)]['search_label'].head(50).tolist()

current_selections = st.session_state['selected_songs']
combined_options = list(set(hits + current_selections))

selected_songs = st.sidebar.multiselect(
    "Select from results:",
    options=combined_options,
    default=current_selections,
    key='final_selection_widget',
    on_change=lambda: st.session_state.update({'selected_songs': st.session_state.final_selection_widget})
)

# Choosing weights
st.sidebar.header("Adjust Algorithm Weights")
sentiment_weight = st.sidebar.slider(
    "Lyrical Sentiment Weight", 
    min_value=0.0, max_value=2.0, value=1.0, step=0.01,
    help="0.0 = Audio Only. 1.0 = Balanced. 2.0 = Lyrics Dominate."
)

# Choosing playlist size
num_recs = st.sidebar.slider("Recommendation Count", 5, 20, 10)


st.title("Spotify Playlist Generator")


if st.sidebar.button("Generate Playlist", type="primary"):
    if not selected_songs:
        st.warning("Please select at least one song to start")
    else:
        # Get seed data
        seed_df = df[df['search_label'].isin(selected_songs)]
        
        # Display seed playlist
        st.subheader("Your Seed Playlist")
        st.dataframe(seed_df[['artists', 'name', 'sentiment']].style.format({"sentiment": "{:.2f}"}), hide_index=True)

        # Calculate target vector
        target_audio = seed_df[AUDIO_FEATURES].mean().values.reshape(1, -1)
        target_sentiment = seed_df[SENTIMENT_FEATURE].mean().values.reshape(1, -1)

        # Get hybrid vector
        all_audio_vectors = df[AUDIO_FEATURES].values
        
        # Apply weight to sentiment vector
        all_sentiment_vectors = df[SENTIMENT_FEATURE].values * sentiment_weight
        target_sentiment_weighted = target_sentiment * sentiment_weight

        final_search_vectors = np.hstack([all_audio_vectors, all_sentiment_vectors])
        final_target_vector = np.hstack([target_audio, target_sentiment_weighted])

        # Cosine similarity to get closest vector to target vector
        sim_scores = cosine_similarity(final_target_vector, final_search_vectors)[0]
        df['similarity'] = sim_scores
        
        # Exclude seed playlist
        results = df[~df['search_label'].isin(selected_songs)]
        
        # Sort results by similarity
        results = results.sort_values(by='similarity', ascending=False)

        # Filter out similar variants (e.g. Radio Edits)
        results = filter_similar_variants(results)

        # Get top recommendations
        results = results.head(num_recs)

        st.divider()
        st.subheader("Generated Playlist")
        
        display_cols = ['artists', 'name', 'similarity', 'sentiment']
        
        # formatting
        st.dataframe(
            results[display_cols].style.format({
                "similarity": "{:.1%}", 
                "sentiment": "{:.2f}"
            }).background_gradient(subset=['similarity'], cmap="Greens"),
            hide_index=True,
            width='stretch'
        )
        
        with st.expander("Debug:"):
            st.write(f"Target Audio Vector (Normalized): {target_audio}")
            st.code(target_audio)
            st.write(f"Target Sentiment (Avg): {target_sentiment[0][0]:.4f}")
            st.write(f"**Applied Weight:** {sentiment_weight}")

else:
    st.info("Select songs, choose sentiment weight, and generate your playlist!")