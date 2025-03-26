import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error

import data_proc as data
import mlp


def load_data(dataset='100k'):
    if dataset == '100k':
        return data.load_100k()
    elif dataset == '20m':
        return data.load_20m()
    raise Exception('Invalid dataset')

def common_interests(id, df):
    uvec = df.loc[id].values[1:]
    df = df.drop(id)
    def combine(row): # bitwise and the genre vector with every other user and sum the result
        ivec = row.values[1:]
        comb = [i & j for (i, j) in zip(uvec, ivec)]
        return sum(comb)

    df['common_genre_count'] = df.apply(combine, axis=1)
    return df['common_genre_count']
    
    
if __name__ == '__main__':
    print("Loading data...")
    df = load_data()

    # # Showing genre distribution
    # genre_count = data.genre_count(df) 
    # plt.bar(genre_count.index, genre_count.values)
    # plt.xlabel('Movie Genres')
    # plt.ylabel('Count')
    # plt.title('Number of Movies in Each Genre')
    # plt.show()

    # # Getting movies and users with more than 50 ratings
    # n_movie_ratings = data.ratings_per_movie(df)
    # n_user_ratings = data.ratings_per_user(df)

    # ratings_50p = n_movie_ratings[n_movie_ratings > 50]
    # users_50p = n_user_ratings[n_user_ratings > 50]

    # print(f"There are {len(ratings_50p)} movies with more than 50 ratings.") 
    # print(f"There are {len(users_50p)} users who rated more than 50 movies.") 

    # # Find users with common genre interests
    # user_genres_df = data.user_genres_watched(df)
    # u_id = int(input('Choose a user ID to see which other users share comment interests: '))
    # shared_int = common_interests(u_id, user_genres_df)
    # com_genre2p = shared_int[shared_int >= 2]
    # print(f"Here are users that share at least two common genres with user {u_id}: ", com_genre2p.index[:10])

    # MLP stuff
    user_input = input("Do you want to train a model or load one? [train/load]: ")
    model = None
    if user_input == 'load':
        try:
            model = joblib.load('../models/initial_mlp.pkl')
        except:
            raise Exception("No model was found, please train one first.")
    elif user_input == 'train':
        print("Performing grid search and k-fold cross validation to train MLP.")
        model = mlp.gscv_model(df)
        print(f"The RMSE using grid search with k-fold cross validation is: {score}.")

        joblib.dump(model, '../models/initial_mlp.pkl')
        print("Model saved at models/initial_mlp.pkl")
    else:
        raise Exception("Invalid answer.")

    X = df.drop(columns=['rating']).values
    y = df['rating'].values
    y_pred = model.predict(X)
    pred_for_score = list(zip(y_pred, y))
    print("The MSE for the model is: ", mean_squared_error(y, y_pred))
    print("Here are the first 20 predictions:")
    for (pred, y) in pred_for_score[:20]:
        print(round(pred), int(y))

