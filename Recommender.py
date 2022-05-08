import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns


class Recommender:

    def __init__(self, movie_data, user_data, model_type, num_recommendations) -> None:
        self.num_recs = num_recommendations
        self.process_data(movie_data, user_data)
        self.train_model(model_type)

    def process_data(self, movie_data, user_data) -> None:
        try:
            self.movies_data = pd.read_csv(movie_data)
            self.user_data = pd.read_csv(user_data)
        except:
            print('CSV load failed')
            return
        
        formatted_data = self.user_data.pivot(index='movieId', columns='userId', values='rating')
        formatted_data.fillna(0, inplace=True)

        csr_data = csr_matrix(formatted_data.values)
        formatted_data.reset_index(inplace=True)
        
        self.all_data = formatted_data
        # self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(csr_data, test_size=0.2)
        self.X_train = csr_data

    def train_model(self, model_type) -> None:
        if model_type == 'knn':
            _metric = 'cosine'
            _algorithm = 'brute'
            num_neighbors = 20
            self.model = NearestNeighbors(metric=_metric, algorithm=_algorithm, n_neighbors=num_neighbors)
        elif model_type == 'xxx':
            pass
        elif model_type == 'yyy':
            pass
        else:
            print('Invalid model type')
            return
        self.model.fit(self.X_train)
        self.model_type = model_type


    def get_recommendations(self, movie_name) -> list:
        if self.model_type == 'knn':
            return self.predict_knn(movie_name)
        elif self.model_type == 'xxx':
            pass
        elif self.model_type == 'yyy':
            pass
        else:
            print('Invalid model type')
            return []

    def predict_knn(self, movie_name) -> list:
        
        movie_id = -1

        for i in range(len(self.movies_data)):
            if self.movies_data.iloc[i].title[:-6].strip() == movie_name:
                movie_id = self.movies_data.iloc[i].movieId
                break
        
        if movie_id != -1:
            movie_idx = self.all_data[self.all_data['movieId'] == movie_id].index[0]
            movie_pt = self.X_train[movie_idx]

            distances, neighbor_idx = self.model.kneighbors(movie_pt, n_neighbors=self.num_recs + 1)
            
            rec_movie_indices = sorted(list(zip(neighbor_idx.squeeze().tolist(), distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
            recommend_frame = []
            for val in rec_movie_indices:
                idx1 = self.all_data.iloc[val[0]]['movieId']
                idx2 = self.movies_data[self.movies_data['movieId'] == idx1].index
                recommend_frame.append({'Title ': self.movies_data.iloc[idx2]['title'].values[0], 'Distance':val[1]})
            df = pd.DataFrame(recommend_frame, index=range(1, self.num_recs + 1))
            return df
        else:
            return "No movies found. Please check your input"