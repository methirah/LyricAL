import pandas as pd
from transformers import pipeline
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Configuration
INPUT_FILE = 'songs_with_attributes_and_lyrics_norm.csv'
OUTPUT_FILE = 'spotify_song_audio_features_norm_250k.csv'
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
SAMPLE_SIZE = 250000
BATCH_SIZE = 64

# torch dataset to avoid sequential pipelines and bottlenecking GPU
class LyricsDataset(Dataset):
    def __init__(self, lyrics_list):
        self.lyrics = lyrics_list
        
    def __len__(self):
        return len(self.lyrics)
        
    def __getitem__(self, i):
        return str(self.lyrics[i])

def run_analysis():
    print(f"Starting optimized build with {SAMPLE_SIZE} songs...")

    # need to have torch gpu version to use GPU
    device = 0 if torch.cuda.is_available() else -1
    device_name = torch.cuda.get_device_name(0) if device == 0 else 'CPU'
    print(f"Device: {device_name}")

    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
        print(f"Downsampled dataset to {len(df)} songs.")

    print("Loading model...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        top_k=None,
        device=device,
        truncation=True,
        max_length=512
    )

    print("Starting inference...")

    lyrics_data = df['lyrics'].fillna("").astype(str).tolist()
    dataset = LyricsDataset(lyrics_data)
    results = []
    error_count = 0

    for i, output in enumerate(tqdm(sentiment_pipeline(dataset, batch_size=BATCH_SIZE), total=len(lyrics_data))):
        try:
            pos = next((x['score'] for x in output if x['label'] == 'positive'), 0.0)
            neg = next((x['score'] for x in output if x['label'] == 'negative'), 0.0)
            norm_score = ((pos - neg) + 1) / 2
            results.append(norm_score)
        except:
            error_count += 1
            results.append(0.5)

    # want to know if any songs are not processed properly because of unsupport language or not
    print(f"Error count: {error_count}")
    df['sentiment'] = results
    final_cols = [c for c in df.columns if c != 'lyrics']
    df[final_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"Final processed dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_analysis()