import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


INPUT_FILE = 'songs_with_attributes_and_lyrics.csv'
OUTPUT_FILE = 'songs_with_attributes_and_lyrics_norm.csv'

AUDIO_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo', 'duration_ms'
]

METADATA_COLS = ['name', 'artists', 'lyrics']

def clean_artist_column(val):
    s = str(val)
    # Remove brackets and quotes (list of strings of artist names
    return s.strip("\"'[]")

def clean_key_column(val):
    # Standard Pitch Class Notation
    key_map = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    
    # Handle as number, otherwise convert to Standard Pitch Class notation
    try:
        num = float(val)

        # -1 indicates no key detected by Spotify        
        if num == -1:
            return np.nan
            
        # Valid key range are 0 to 11 (Standard Pitch Class notation)
        if 0.0 <= num <= 11.0:
            return num
            
    except (ValueError, TypeError):
        pass

    s_val = str(val).strip().title() 
    
    if s_val in key_map:
        return float(key_map[s_val])
    
    return np.nan

def clean_mode_column(val):
    # Handle as number, otherwise map to designated mode (Major = 1, Minor = 0)
    try:
        num = float(val)
        return 1.0 if num == 1.0 else 0.0
    except (ValueError, TypeError):
        pass

    s_val = str(val).strip().lower()
    if 'major' in s_val: return 1.0
    if 'minor' in s_val: return 0.0
    
    return np.nan


def preprocess_data():
    print("Preprocessing...")
    
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Original Dataset Size: {len(df)} rows")
    except FileNotFoundError:
        print(f"ERROR: Could not find input file: '{INPUT_FILE}'.")
        return
    
    # Clean Features
    print("Cleaning columns...")
    df['artists'] = df['artists'].apply(clean_artist_column)
    df['key'] = df['key'].apply(clean_key_column)
    df['mode'] = df['mode'].apply(clean_mode_column)
    
    # Drop NaN/Null Columns
    df = df.dropna(subset=['name', 'artists'])   
    print(f"Size after removing null name/artist cols: {len(df)} rows") 
    df = df.dropna(subset=['lyrics'])
    df = df[df['lyrics'].str.strip().astype(bool)]    
    print(f"Size after removing empty/null lyrics: {len(df)} rows")
    df = df.dropna(subset=AUDIO_FEATURES)
    print(f"Size after removing null audio feature cols: {len(df)} rows")
    
    # Drop duplicates
    df = df.drop_duplicates(subset=['artists', 'name'])
    print(f"Size after removing duplicate songs: {len(df)} rows")
    
    # Normalize Features (preparing for cosine similarity)
    print("Normalizing Features...")
    scaler = MinMaxScaler()
    cols_to_normalize = ['tempo', 'loudness', 'duration_ms', 'key']
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    
    final_cols = ['id'] + METADATA_COLS + AUDIO_FEATURES
    final_cols = [c for c in final_cols if c in df.columns]
    
    df_norm = df[final_cols]
    df_norm.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Cleaned dataset saved to: {OUTPUT_FILE}")
    print(f"Final Row Count: {len(df_norm)}")

if __name__ == "__main__":
    preprocess_data()