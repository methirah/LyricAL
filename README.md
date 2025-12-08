# AI Playlist Generator (Group 119)

### 1. Installation
Ensure you have Python 3.8+ installed.

1. **Create/Activate Virtual Environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   # Install requirements
    pip install -r requirements.txt
   pip install torch --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118) # for gpu version

2. **Run the app:**
    ```bash
    streamlit run app.py
    ```
    In order to change the database, simply change the DATA_FILE variable to either spotify_song_audio_features_norm_25k.csv, spotify_song_audio_features_norm_250k.csv, or spotify_song_audio_features_norm_vader.csv. You can also create your own database using roberta by changing the sample size and creating a new output file name. Then after it's finished running, simply change the DATA_FILE variable to the new .csv that you created and run the app.