# MoodTunes: A Mood-based Song Recommendation System 🎵

## Table of Contents
1. [Project Description](#project-description)
2. [Features](#features)
3. [Data](#data)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results & Challenges](#results-&-challenges)
7. [Future Improvements](#future-improvements)
8. [Acknowledgments](#acknowledgement)

## Project Description

This project aims to build a mood-based song recommendation system that suggests songs based on the user’s emotions and thoughts. Unlike existing solutions that rely on audio features, this system analyses song lyrics to capture emotional depth and thematic relevance.

The recommendation process incorporates **two key elements**:
- **Emotion Analysis** – Identifies the dominant emotion of a song using a pre-trained BERT-based emotion classification model.
- **Topic Modelling** – Extracts key themes from lyrics using Latent Dirichlet Allocation (LDA) to match user prompts with thematically relevant songs.

## Features

These are the features of this app:
- **User Input Processing**: Takes a text-based user prompt describing their feelings and/or situation.
- **Emotion Detection**: Classifies emotions in lyrics into six categories using a fine-tuned BERT model.
- **Topic Modelling with LDA**: Identifies hidden themes in lyrics to improve song relevance.
- **Recommendation Engine**: Matches the user’s input with the most similar songs based on its emotion and topic similarity.

## Data

### Data Source
The original data was downloaded from Kaggle: [960K Spotify Songs With Lyrics data](https://www.kaggle.com/datasets/bwandowando/spotify-songs-with-attributes-and-lyrics). This contains a variety of features including song metadata, audio features and lyrics. However, Spotify's TOS prevents use of their audio features and data for machine learning. Therefore, only the song title, album title, artist(s) name and song lyrics were kept as they are publically available. 

### Data processing
Due to the large sample size, only 10,000 randomly sampled songs were used. For each song, their lyrics were used for emotion detection using a [pre-trained BERT model for emotion classification](https://huggingface.co/michellejieli/emotion_text_classifier); and for LDA topic modelling. The final dataset, provided in the data folder, contains each songs, its metadata, its dominant emotion and topic distribution. 

Refer to `song_recommendation.ipynb` for full details on the processing.
    
## Installation
The app can be run either locally or using the web-browser.

**To run this app locally, follow these steps:**
1. Open your terminal/command prompt.
2. Navigate to your desired directory.
3. Git clone this repository:
   ```bash
   git clone https://github.com/IsabelleRaj/Mood-Based-Song-Recommendation
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Launch the streamlit app:
   ```bash
   streamlit run song_recommendation_app.py.py
   ```

**To launch the deployed app:**
1. Visit this link: [MoodTunes: Mood-based Song Recommendation](https://mood-based-song-recommendation-isabelleraj.streamlit.app/). 

## Usage
Here is a demonstration of the app:

<img src="MoodTunes Demo Video.gif" width="1000" height="450">

## Results & Challenges
Here are some key findings and challenges:
1. Most songs in the dataset were labelled as joyful or sad; and also belong to the the love & relationships dominant topic, aligning with common emotional expressions in music but leading to an imbalanced dataset.
2. Low coherence scores in LDA suggested that lyrics may not always have strong, well-defined topics.
3. Human evaluation was necessary to validate the accuracy of emotion predictions.

## Future Improvements
Here are some future improvements:
1. Expand emotion categories for more nuanced recommendations (more than just the 6 Ekman emotions).
2. Improve topic coherence by experimenting with different modelling techniques e.g., keyword matching.
3. Use of genres to obtain a more balanced dataset.

## Acknowledgments
This project was written by me ([@IsabelleRaj](https://github.com/IsabelleRaj)), as part of the Digital Futures Academy. The app was created and deployed using [Streamlit](https://streamlit.io/).
