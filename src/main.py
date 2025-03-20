import pandas as pd

if __name__ == '__main__':
    # Load and prepare data
    print("Loading data...")
    movie_df = pd.read_csv('res/movie.csv').drop(columns=['title'])
    ratings_df = pd.read_csv('res/rating.csv').drop(columns=['timestamp'])

    df = pd.merge(movie_df, ratings_df, on='movieId')
    df = df.drop_duplicates()

    genre_count = df.groupby('genres').size().reset_index(name='count').sort_values('count', ascending=False)
    print(genre_count)

    # TODO: bar chart with genre_count

    rating_count = df.groupby('movieId').size().reset_index(name='count')
    user_count = df.groupby('userId').size().reset_index(name='count')
    
    movie_50_ratings = rating_count[rating_count['count'] > 50]
    user_50_ratings = user_count[user_count['count'] > 50]

    print("Movies with more than 50 ratings: ", len(movie_50_ratings))
    print("Users who rated more than 50 movies: ", len(user_50_ratings))

    # TODO Find users with common interests

    ## PART 4: MLP ##
