# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer 
from nltk.stem import SnowballStemmer
from word2number import w2n
import re
import string
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Download the stopwords
nltk.download('stopwords')

# Set the app title
st.title('MoodTunes: A Mood-Based Song Recommendation System :musical_note:')

# Display the app introduction
st.markdown("""
###### By Isabelle Rajendiran
            
Music has this amazing way of connecting with our emotions. Whether you're feeling down, excited or anywhere in-between, there is always a song to fit your current mood.

Just type in a few words about your emotions or the kind of song you're looking for, and I'll recommend you some songs based on this!
""")

## Load the dataset used for the song recommendation
song_rec_df = pd.read_csv('Data/song_recommendation.csv')

## Load the BERT model chosen to detect the emotion
michelle = pipeline(task="text-classification", model="michellejieli/emotion_text_classifier", top_k=1) # top_k=1 to get only the top emotion

## load the dictionary and LDA model used for topic modelling
dictionary = Dictionary.load('Model/gensim_dictionary.dict')
lda_model = LdaMulticore.load('Model/gensim_lda_model.model')

## Initialise a stemmer to get to the root word
stemmer = SnowballStemmer(language='english')

## Initalise the tokeniser
tokeniser = WhitespaceTokenizer() # Extracts the tokens without whitespaces, new line and tabs 

## Create a list of stopwords and punctuation
remove = stopwords.words('english')
remove.extend(string.punctuation) # Extend the list with punctuation

## Extend the remove list with common contractions
contractions = ["they'r", "they'v", "can't", "won't", "don't", "i'm", "it's", "i'll", "ain't", "can't", "won't", "don't", "i'm", "it's", "he's", "she's"]
remove.extend(contractions)

def detect_prompt_emotion(prompt):
    """
    Detect the prompt's emotion using the BERT model.

    Parameters
    ----------
    prompt : string
        A text string that you want to detect the emotion for

    Returns
    -------
    The emotion of the prompt as a string
    """  
    # Perform quick, basic cleaning on the prompt
    prompt = prompt.strip().lower().replace('\n', '')

    # Predict the emotion
    try:
        emotion = michelle(prompt)
    except RuntimeError: # Occurs if the length of the prompt exceeds limit (though unlikely)
        print("Sorry! That was too long. Could you try shortening your entry?")

    # Return just the label
    return emotion[0][0]['label']

def preprocess_prompt(prompt):
    """ 
    Cleans up a user prompt for retrieving the topic distribution

    Parameters
    ----------
    prompt : string
        A text string that you want to clean

    Returns
    -------
    The same string but cleaned up, stemmed and tokenised (as a list)
    """  
    # Apply pre-token cleaning
    prompt = prompt.lower().strip()
    
    # Tokenise the prompt using the whitespace tokeniser
    prompt_tokens = tokeniser.tokenize(prompt)

    # Apply post-token cleaning
    prompt_tokens = [word.strip(string.punctuation) for word in prompt_tokens]
    prompt_tokens = [word for word in prompt_tokens if word not in remove] # Remove stopwords and punctuation

    
    prompt_tokens_clean = []
    for word in prompt_tokens:
        try:
            word = str(w2n.word_to_num(word)) # Convert numbers which are written as text e.g. ten -> 10
        except:
            word = word
        prompt_tokens_clean.append(word)
        
    prompt_tokens_clean = [re.sub(r'\d+', '', word) for word in prompt_tokens_clean] # Remove numbers
    
    prompt_tokens_clean = [stemmer.stem(word) for word in prompt_tokens_clean] # Apply the stemmer
    prompt_tokens_clean = [word for word in prompt_tokens_clean if word not in remove] # In case some words became stopwords

    return prompt_tokens_clean

def get_topic_distribution(prompt):
    """
    Retrieve the topic distribution of the user prompt

    Parameters
    ----------
    prompt : string
        A text string that you want get the topic distribution for

    Returns
    -------
    A list of tuples containing the topic number and its distribution for each topic
    """
    # Preprocess the user prompt
    tokens = preprocess_prompt(prompt)
    
    # Convert the token to  a Bag of Words (BoW) using the dictionary
    bow = dictionary.doc2bow(tokens)
    
    # Get the topic distribution (inference on new text)
    topic_distribution = lda_model.get_document_topics(bow)
    
    return topic_distribution 

def process_user_prompt(prompt):
    """
    Return the calculated emotion and topic distribution of the prompt

    Parameters
    ----------
    prompt : string
        A text string

    Returns
    -------
    A tuple containing the emotion and topic distribution
    """
    # Detect emotion from the user prompt
    emotion = detect_prompt_emotion(prompt)
    
    # Get the topic distribution of the user prompt
    topic_distribution = get_topic_distribution(prompt)
    
    return emotion, topic_distribution

def match_emotion(user_emotion, song_df):
    """
    Filter the dataframe for matching emotion to the user's prompt

    Parameters
    ----------
    user_emotion : string
        A text string containing the emotion
    song_data: DataFrame
        A DataFrame containing the song information

    Returns
    -------
    A DataFrame containing the filtered song information
    """
    # Filter the dataframe for only songs with the same emotion
    song_df_filtered = song_df[song_df['emotion'] == user_emotion].copy()
    
    return song_df_filtered

## Create a function to compute cosine similarity between the user's topic distribution and each song's topic distribution

def calculate_cosine_similarity(user_topic_dist, song_df):
    """
    Compute cosine similarity between the user's topic distribution 
    and each song's topic distribution

    Parameters
    ----------
    user_topic_dist : list
        A list of tuples containing the topic distribution
    song_df: DataFrame
        A DataFrame containing the song information

    Returns
    -------
    An array containing the similarity scores for each song in the DataFrame
    """
    # Convert user topic distribution to a numpy array
    user_topic_array = np.array([x[1] for x in user_topic_dist]) # x[1] gives the actual value rather than topic number e.g., 0.2 (20%)

    # Filter the the song dataframe and convert the topic distribution to a numpy array
    song_topic_cols = [col for col in song_df.columns if col.startswith('Topic_distr')] # Extract the columns
    song_topic_array = song_df[song_topic_cols].to_numpy() # Turn into a numpy array
    
    # Compute cosine similarity using the user topic array and already exisiting song topic array
    similarities = cosine_similarity([user_topic_array], song_topic_array)

    # Return the similarity scores with each song 
    return similarities[0] # Returns a list with length 10,000 for each song for my dataset

def recommend_songs(prompt):
    """
    Recommend top 5 songs based on matching emotion and topic similarity.

    Parameters
    ----------
    prompt : string
        A text string containing the user's prompt

    Returns
    -------
    A DataFrame containing the top 5 song recommendation and the emotion as a string
    """
    # Compute the emotion and topic distribution of the user's prompt
    emotion, topic_dist = process_user_prompt(prompt)
   
    # Filter the song datasets based on maching emotion
    emotion_matched_songs_df = match_emotion(emotion, song_rec_df) # This is where the original dataframe goes into

    # Calculate the cosine similarity based on topic distribution for only the emotion matched songs
    similarity_scores = calculate_cosine_similarity(topic_dist, emotion_matched_songs_df)

    # Add the similarity scores to the dataframe
    emotion_matched_songs_df['similarity_score'] = similarity_scores 

    # Get the top 5 most similar songs
    top_5_songs = emotion_matched_songs_df.sort_values(by='similarity_score', ascending=False).head(5).reset_index(drop = True)
    
    # Clean the dataframe and add a rank
    top_5_songs.rename(columns={'name': 'Song Title', 'artists': 'Artists'}, inplace = True)
    top_5_songs['Rank'] = top_5_songs.index + 1
 
    # Return the important information
    return emotion, top_5_songs[['Rank', 'Song Title', 'Artists']]

## Ask for the user's prompt
user_prompt = st.text_area("**Tell me how you're feeling, and let’s find your perfect song! :notes:**")
button=st.button("Hit me", type="primary")

if button == True:
    # Check if the prompt is empty
    if len(user_prompt) == 0:
        st.write(":red[It doesn't look like you've typed anything in. Try again.]" )
    else:
        ## Run the recommendation on the prompt
        emotion, top_5 = recommend_songs(user_prompt)
        
        ## Display the emotion detected
        if emotion != 'neutral':
            st.markdown(f"It looks like you are experiencing **{emotion}**. Let me find some recommendations to suit this.")
        else:
            st.markdown(f"It doesn't seem like you're experiencing any strong emotion. Let me find some recommendations to suit this.")
    
        ## Display the results as a dataframe
        st.dataframe(top_5, hide_index= True)

if len(user_prompt) != 0:
    ## Ask for feedback on recommendation
    st.markdown("How many stars would you rate the recommendations?")
    star_mapping = ["one", "two", "three", "four", "five"]
        
    selected = st.feedback("stars")
    # If the feedback is not empty
    if selected is not None:
        st.markdown(f"You selected {star_mapping[selected]} star(s). Thank you for your feedback! :smile:")

## In the future, add some functionalities to store the ratings with each prompt and recommendations.

# Add sidebar content
with st.sidebar:
    # Give it a title
    st.title(":red[Mood-based Song Recommendation]")

    # Short description
    st.markdown("""
    Find the perfect song based on how you're feeling and your current situation.\n
    """)

    # How to use the app
    st.subheader("How to Use")
    st.markdown(
        """
        1. **Describe how you're feeling** in the text box.
        2. **Click 'Hit me'** to get song recommendations.
        3. **See your results** and enjoy the music!
        4. **Give feedback** by rating the recommendations.
        """
    )

    # Example prompts
    st.subheader("Example Prompts")
    st.markdown(
        """
        - *"I'm feeling nostalgic about my childhood."*
        - *"I just got some great news and want to celebrate!"*
        - *"I need a song to help me get through a tough day."*
        - *"Looking for something calm to help me relax."*
        """
    )

    # How does the app word
    st.subheader("How Does it Work")
    st.markdown(
        """
    - **Emotion Detection:** The app analyses your input to determine your emotion using a transformer-based BERT model.
    - **Topic Detection:** It identifies the main themes in your input using topic modelling.
    - **Similarity Matching:** It calculates a similarity score between your input and songs based on their lyrics.
    - **Smart Recommendations:** It suggests songs that best match your emotions and themes.
        """
    )

    # Add a link to my GitHub
    st.markdown(
        "For more information on this app, refer to my [GitHub](https://github.com/IsabelleRaj/Mood-Based-Song-Recommendation).",
        unsafe_allow_html=True
    )
