# ## Final Assessment - BCI3333 Machine Learning Application
## Noel Foo Sei Wee CB20066
# spotify_all_genres_tracks.csv (https://www.kaggle.com/datasets/laurabarreda/spotify-tracks-by-genre-8-genres-classification)

import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Title and description
st.title('Spotify Tracks by Genre')
st.write('A dataset containing 9199 tracks belonging to Spotify playlists. The tracks are classified with the genre of the playlist, posted by Spotify or relevant agencies from the music industry.')

st.write("""
        Features: 
        - track_popularity: int, from 0 to 100, how popular the track is
        - artist_popularity: int, from 0 to 100, how popular the artist is
        - danceability: float, from 0.0 to 1.0, the suitability of the track for dancing
        - energy: float, from 0.0 to 1.0, the intensity and activity of the track
        - loudness: float, from -60.0 to 2.0, the overall loudness of the track in dB
        - mode: int, 0 or 1, the modality of the track, 0 = Major, 1 = Minor
        - acousticness: float, from 0.0 to 1.0, the confidence of the track being acoustic
        - valence: float, from 0.0 to 1.0, musical positiveness. Higer values mean more positivity
        - tempo: float, from 0.0 to 250.0, overall tempo of the track in BPM.
        - time_signature: int, from 0 to 5, overall time signature of the track, how many beats are in each bar.
        """)

# # Background image
# page_bg_img = '''
# <style>
# .stApp {
# background-image: url("https://images.unsplash.com/photo-1535925191244-17536ca4f8b6?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8c3BvdGlmeSUyMG11c2ljfGVufDB8fDB8fHww&w=1000&q=80");
# background-size: auto;
# }
# </style>
# '''

# st.markdown(page_bg_img, unsafe_allow_html=True)


# Sidebar - User input features
st.sidebar.header('User Input Features')

# Collects user input features
# track_popularity,artist_popularity,danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms,time_signature   
def user_input_features():
    track_popularity = st.sidebar.slider('Track Popularity', 0, 100, 93, 1)
    artist_popularity = st.sidebar.slider('Artist Popularity', 0, 100, 94, 1)
    danceability = st.sidebar.slider('Danceability', 0.0, 1.0, 0.7, 0.01)
    energy = st.sidebar.slider('Energy', 0.0, 1.0, 0.35, 0.01)
    # key: 0 = C, 1 = C♯/D♭, 2 = D, and so on
    # slider shows the key in letter but the value is in number
    # key = st.sidebar.selectbox('Key',('C','C♯/D♭','D','D♯/E♭','E','F','F♯/G♭','G','G♯/A♭','A','A♯/B♭','B'))
    # # make the key into number start from 0 using dictionary
    # key_dict = {'C':0,'C♯/D♭':1,'D':2,'D♯/E♭':3,'E':4,'F':5,'F♯/G♭':6,'G':7,'G♯/A♭':8,'A':9,'A♯/B♭':10,'B':11}
    # key = key_dict[key]
    # mode: 0 = Major, 1 = Minor
    mode = st.sidebar.selectbox('Mode',('Major','Minor'))
    # make the mode into number start from 0 using dictionary
    mode_dict = {'Major':0,'Minor':1}
    mode = mode_dict[mode]
    loudness = st.sidebar.slider('Loudness', -60.0, 2.0, -22.30, 0.1)
    # speechiness = st.sidebar.slider('Speechiness', 0.0, 1.0, 0.31, 0.01)
    acousticness = st.sidebar.slider('Acousticness', 0.0, 1.0, 1.0, 0.01)
    # instrumentalness = st.sidebar.slider('Instrumentalness', 0.0, 1.0, 1.0, 0.01)
    # liveness = st.sidebar.slider('Liveness', 0.0, 1.0, 1.0, 0.01)
    valence = st.sidebar.slider('Valence', 0.0, 1.0, 0.64, 0.01)
    tempo = st.sidebar.slider('Tempo', 0.0, 250.0, 94.60, 0.1)
    # duration_ms = st.sidebar.slider('Duration (ms)', 0, 1000000, 500000, 1000)
    time_signature = st.sidebar.slider('Time Signature', 0, 5, 2, 1)

    data = {
            'track_popularity': track_popularity,
            'artist_popularity': artist_popularity,
            'danceability': danceability,
            'energy': energy,
            # 'key': key,
            'mode': mode,
            'loudness': loudness,
            # 'speechiness': speechiness,
            'acousticness': acousticness,
            # 'instrumentalness': instrumentalness,
            # 'liveness': liveness,
            'valence': valence,
            'tempo': tempo,
            # 'duration_ms': duration_ms,
            'time_signature': time_signature
            }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Read Spotify dataset
spotify_raw = pd.read_csv('spotify_all_genres_tracks.csv')

# drop duplicate rows
spotify_raw.drop_duplicates(inplace=True)

# drop rows with missing values
spotify_raw.dropna(inplace=True)

# drop columns that are not needed
spotify = spotify_raw.drop(columns=['track_id', 'playlist_url', 'playlist_name', 'track_name', 'artist_name', 'album', 'album_cover', 'artist_genres', 'genre', 'liveness', 'key', 'duration_ms', 'instrumentalness', 'speechiness'])

# combine the user input features and the spotify dataset
df = pd.concat([input_df,spotify],axis=0)

# select only the first row (the user input data)
df = df[:1]

# load the saved model
load_clf = pickle.load(open('spotify_rf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)

# Display prediction
st.subheader('Prediction')

# Predicted output: blues = 0, classical = 1, jazz = 2, hiphop = 3, pop = 4, reggae = 5, rock = 6, electronic = 7
case = {0:'blues', 1:'classical', 2:'jazz', 3:'hiphop', 4:'pop', 5:'reggae', 6:'rock', 7:'electronic'}
st.write('The predicted genre is', '**' + case[prediction[0]].capitalize() + '**')