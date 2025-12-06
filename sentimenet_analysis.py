import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm


INPUT_FILE = 'songs_with_attributes_and_lyrics_norm.csv' 
OUTPUT_FILE = 'spotify_song_audio_features_norm.csv'

def run_sentiment_analysis():
    print("Starting Sentiment Analysis...")

    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} songs.")
    except FileNotFoundError:
        print(f"ERROR: Could not find input file: '{INPUT_FILE}'.")
        return

    # Download lexicons
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("Downloading VADER lexicon...")
        nltk.download('vader_lexicon')

    sid = SentimentIntensityAnalyzer()

    
    print("Analyzing lyrics...")
    
    # use tqdm as progress bar, cool :D
    results = []
    for text in tqdm(df['lyrics'], desc="Processing Lyrics"):
        # 'compound' score (-1.0 to +1.0)
        raw_score = sid.polarity_scores(str(text))['compound']
        norm_score = (raw_score + 1) / 2
        results.append(norm_score)

    # Add sentiment feature, drop old lyric feature
    df['sentiment'] = results
    df = df.drop(columns=['lyrics']) 


    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Final processed dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_sentiment_analysis()