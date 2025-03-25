import pandas as pd
import matplotlib.pyplot as plt
import data_proc as data


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

