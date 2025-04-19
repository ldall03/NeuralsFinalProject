import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances, euclidean_distances

from data_proc import GENRES

class CBModel():
    def __init__(self, df, n_components=200):
        self.n_components = n_components
        self.df = df.drop_duplicates(subset=['user_id', 'movie_id'])
        self.user_movie_matrix = self.df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
        self.user_factors = None
        self.movie_factors = None
        self.pred = None
        self.mse = -1

    def fit(self):
        svd = TruncatedSVD(n_components=self.n_components)
        self.user_factors = svd.fit_transform(self.user_movie_matrix)
        self.movie_factors = svd.components_
        self.pred = np.dot(self.user_factors, self.movie_factors)

    def predict(self, df):
        u_ids = df['user_id'].values
        m_ids = df['movie_id'].values
        ratings = df['rating'].values
        errors = []
        y_pred = []

        for (u, m, y) in zip(u_ids, m_ids, ratings):
            u_id = self.user_movie_matrix.index.get_loc(u)
            m_id = self.user_movie_matrix.columns.get_loc(m)

            p = self.pred[u_id, m_id]
            y_pred.append(p)
            errors.append((y - p)**2)

        arr = np.array(errors)
        self.mse = np.mean(arr)
        return y_pred

def combine(block, t):
    weighted_avg = np.average(block.iloc[:, 3:].values, axis=0, weights=block['rating'])
    vec = [1 if v > t else 0 for v in weighted_avg]
    ret = pd.Series(vec, block.columns[3:])
    return ret


class CBFModel():
    def __init__(self, df, decision_threshold=0.2):
        self.dec_t = decision_threshold
        self.pred = None
        self.mse = -1
        self.df = df

    def fit(self):
        u_prof = self.df.groupby("user_id")
        u_prof = u_prof.apply(combine, self.dec_t)
        mov_prof = self.df.sort_values('movie_id').drop(columns=['user_id', 'rating']).drop_duplicates(subset='movie_id').set_index('movie_id')

        # pred = cosine_similarity(u_prof, mov_prof)
        pred = pairwise_distances(u_prof.values, mov_prof.values, metric='jaccard')
        pred = 1 + 4 * (pred - pred.min()) / (pred.max() - pred.min()) # Normalize between 1 and 5
        self.pred = pd.DataFrame(pred, index=u_prof.index, columns=mov_prof.index)

    def predict(self, df):
        u_ids = df['user_id'].values
        m_ids = df['movie_id'].values
        ratings = df['rating'].values
        errors = []
        y_pred = []

        for (u, m, y) in zip(u_ids, m_ids, ratings):
            p = self.pred.loc[u, m]
            y_pred.append(p)
            errors.append((y - p)**2)

        arr = np.array(errors)
        self.mse = np.mean(arr)
        return y_pred

