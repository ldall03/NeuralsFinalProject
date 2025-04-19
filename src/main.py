import random
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score

import data_proc as data
import mlp
import hybrid_model as hbm
import GA as ga


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

    # Showing genre distribution
    genre_count = data.genre_count(df) 
    plt.bar(genre_count.index, genre_count.values)
    plt.xlabel('Movie Genres')
    plt.ylabel('Count')
    plt.title('Number of Movies in Each Genre')
    plt.show()

    # Getting movies and users with more than 50 ratings
    n_movie_ratings = data.ratings_per_movie(df)
    n_user_ratings = data.ratings_per_user(df)

    ratings_50p = n_movie_ratings[n_movie_ratings > 50]
    users_50p = n_user_ratings[n_user_ratings > 50]

    print(f"There are {len(ratings_50p)} movies with more than 50 ratings.") 
    print(f"There are {len(users_50p)} users who rated more than 50 movies.") 

    # Find users with common genre interests
    user_genres_df = data.user_genres_watched(df)
    u_id = int(input('Choose a user ID to see which other users share comment interests: '))
    shared_int = common_interests(u_id, user_genres_df)
    com_genre2p = shared_int[shared_int >= 2]
    print(f"Here are users that share at least two common genres with user {u_id}: ", com_genre2p.index[:10])

    user_input = input("Do you want to train a model or load one? [train/load]: ")
    model = None

    train_df, test_df = data.train_test_split_df(df, size=0.2, random_state=42)
    cbf_model = hbm.CBFModel(train_df, decision_threshold=0.005)
    cbf_model.fit()
    cb_svd = hbm.CBModel(train_df, n_components=700)
    cb_svd.fit()
    cbcf_train_df = data.create_cbcf_df(cbf_model, cb_svd, train_df)
    cbcf_test_df = data.create_cbcf_df(cbf_model, cb_svd, test_df)

    if user_input == 'load':
        try:
            model = joblib.load('../models/hybrid_mlp.pkl')
        except:
            raise Exception("No model was found, please train one first.")
    elif user_input == 'train':
        print("Performing grid search and k-fold cross validation to train MLP.")

        hybrid_model = mlp.cbcf_mlp(cbcf_train_df)

        X_test = cbcf_test_df.drop(columns=['rating']).values
        y_test = cbcf_test_df['rating'].values

        pred = hybrid_model.predict(X_test)
        acc = accuracy_score(pred, y_test)
        print(acc)

        joblib.dump(hybrid_model, '../models/hybrid_mlp.pkl')
        print("Model saved at models/initial_mlp.pkl")
        model = hybrid_model
    else:
        raise Exception("Invalid answer.")

    X = cbcf_test_df.drop(columns=['rating']).values
    y = cbcf_test_df['rating'].values
    y_pred = model.predict(X)
    pred_for_score = list(zip(y_pred, y))
    print("The accuracy for the model is: ", mean_squared_error(y, y_pred))
    print("Here are 20 random predictions:")
    for (pred, y) in random.sample(pred_for_score, 20):
        print(round(pred, 2), int(y))

    # GA
    mlp_ga_instance = ga.MLP_GA()
    mlp_ga_instance.init_ga(df)
    best_params = mlp_ga_instance.run_ga()
    ga_mlp = mlp_ga_instance.train_best_mlp()
    joblib.dump(ga_mlp, '../models/ga_mlp.pkl')
    print("Model saved at models/ga_mlp.pkl")

    

