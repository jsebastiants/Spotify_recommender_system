import pickle
import spotipy 
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import dill
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import json

app = Flask(__name__)

#Load the model
song_cluster_pipe = pickle.load(open('km.pkl', 'rb'))

# Load the data
data = pd.read_csv('C:/Users/Sebastian/Desktop/Spotify_data_2.csv', sep=';')

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='016b17bda24c49b597da608bfcbc8475',
                                                           client_secret='3c14685657434148890cf54d96928442'))

number_cols = ['popularity', 'acousticness', 'danceability', 'duration_ms', 'energy', 'liveness',
    'loudness', 'mode', 'speechiness', 'instrumentalness', 'tempo', 'valence']


class Spotify_recommender:

    def find_song(self, name, artist):
        song_data = defaultdict()
        results = sp.search(q='track:{} artist:{}'.format(name, artist), type='track', limit=1)
        if results['tracks']['items'] == []:
            return None
        
        results = results['tracks']['items'][0]
        track_id = results['id']
        audio_features = sp.audio_features(track_id)[0]
        
        song_data['name'] = [name]
        song_data['artist'] = [artist]
        song_data['duration_ms'] = [results['duration_ms']]
        song_data['popularity'] = [results['popularity']]
        
        for key, i in audio_features.items():
            song_data[key] = i
            
        return pd.DataFrame(song_data) 

    def get_song_data(self, song, spotify_data):
    
        try:
            song_data = spotify_data[(spotify_data['song_names'] == song['name'])].iloc[0]
            return song_data
        except IndexError:
            return Spotify_recommender.find_song(self, song['name'], song['artist'])
    
    def get_mean_vector(self, song_list, spotify_data):
        
        song_vectors = []
        for song in song_list:
            song_data = Spotify_recommender.get_song_data(self, song, spotify_data)
            if song_data is None:
                print('Warning: {} does not exist in Spotify or in database, gil'.format(song['name']))
                continue
            song_vector = song_data[number_cols].values
            song_vectors.append(song_vector)
            
        song_matrix = np.array(list(song_vectors))
        return np.mean(song_matrix, axis = 0)

    def flatten_dict_list(self, dict_list):
        
        flattened_dict = defaultdict()
        for key in dict_list[0].keys():
            flattened_dict[key] = []
            
        for dictionary in dict_list:
            for key, i in dictionary.items():
                flattened_dict[key].append(i)
        
        return flattened_dict

    def recommend_songs(self, song_list, spotify_data, n_songs=10):
        
        metadata_cols = ['song_names']
        song_dict = Spotify_recommender.flatten_dict_list(self, song_list)
        
        song_center = Spotify_recommender.get_mean_vector(self, song_list, spotify_data)
        scaler = song_cluster_pipe.steps[0][1]
        scaled_data = scaler.transform(spotify_data[number_cols])
        scaled_song_center = scaler.transform(song_center.reshape(1,-1))
        distances = cdist(scaled_song_center, scaled_data, 'cosine')
        index = list(np.argsort(distances)[:, :n_songs][0])
        
        rec_songs = spotify_data.iloc[index]
        rec_songs = rec_songs[~rec_songs['song_names'].isin(song_dict['name'])]
        return rec_songs[metadata_cols].to_dict(orient='records')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    # Here we introduced the data in the void gaps
    data_request =[str(x) for x in request.form.values()]
    d = {}
    d_list = []
    for i in data_request:
        d['name'] = i
        d_list.append(d)
        print(d_list)
    #print(data_request)
    #song_name = str(data_request[0])
    #print(song_name)
    
    # Now, we have to search the song in the database,
    # furthermore  we have to get recommendations by the Spotify API
    
    my_recommendations = Spotify_recommender()
    result = my_recommendations.recommend_songs(d_list, data)
    print(result)
    #return render_template("home.html",prediction_text="The recommendation is {}".format(result))
    return result

if __name__=="__main__":
    app.run(debug=True)