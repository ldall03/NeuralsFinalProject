import chardet
import pandas as pd

DATA_COLUMNS = [
    "movie_id", "movie_title", "release_date", "video_release_date",
    "imdb_url", "unknown", "action", "adventure", "animation",
    "children", "comedy", "crime", "documentary", "drama", "fantasy",
    "film-noir", "horror", "musical", "mystery", "romance", "sci-fi",
    "thriller", "war", "western"
]

SELECTED_COLUMNS = [
    "user_id", "movie_id", "rating", "unknown", "action", "adventure", 
    "animation", "children", "comedy", "crime", "documentary", "drama", 
    "fantasy", "film-noir", "horror", "musical", "mystery", "romance", 
    "sci-fi", "thriller", "war", "western"
]

GENRES = [
    "unknown", "action", "adventure", "animation", "children", "comedy", 
    "crime", "documentary", "drama", "fantasy", "film-noir", "horror", 
    "musical", "mystery", "romance", "sci-fi", "thriller", "war", "western"
]

def detect_encoding():
    with open("../res/100k_ds/u.item", "rb") as f:
        result = chardet.detect(f.read(100000))  # Read first 100k bytes
    print(result["encoding"])  # Print detected encoding


def load_20m():
    movie_df = pd.read_csv('../res/20m_ds/movie.csv', header=0, names=['movie_id', 'title', 'genres'])
    ratings_df = pd.read_csv('../res/20m_ds/rating.csv', header=0, names=['user_id', 'movie_id', 'rating', 'timestamp'])

    df = pd.merge(movie_df, ratings_df, on='movie_id')
    df = df[["user_id", "movie_id", "rating", "genres"]]
    df = df.drop_duplicates()

    # Transform genres column to genre matrix
    for col in GENRES: # Add new col for each genre
        df[col] = 0

    def pop_genre(row):
        genres = row['genres'].split('|')
        for g in genres:
            row[g.lower()] = 1
        return row
    df = df.apply(pop_genre, axis=1)
    df = df.drop(columns=['genres'])
    print(df.columns)

    return df

def load_100k():
    ratings_df = pd.read_csv('../res/100k_ds/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'], encoding='ISO-8859-1') 
    movies_df = pd.read_csv('../res/100k_ds/u.item', sep='|', names=DATA_COLUMNS, encoding='ISO-8859-1') 

    # Convert pk's to ints otherwise we get type errors on the merge
    ratings_df['movie_id'] = ratings_df['movie_id'].astype(int)
    movies_df['movie_id'] = ratings_df['movie_id'].astype(int)

    df = pd.merge(movies_df, ratings_df, on='movie_id')

    df = df[SELECTED_COLUMNS]
    df = df.drop_duplicates()
    return df

def genre_count(df):
    genres_df = df[GENRES]
    return genres_df.sum()

def ratings_per_movie(df):
    count_df = df.groupby('movie_id').count()
    return count_df['rating']

def ratings_per_user(df):
    count_df = df.groupby('user_id').count()
    return count_df['rating']

def user_genres_watched(df):
    user_genre_df = df.groupby('user_id').max()[GENRES]
    return user_genre_df

def rating_analysis(df):
    l = len(df)
    print("Rating mean: ", df['rating'].mean())
    print("Rating median: ", df['rating'].median())
    rating_count = df.groupby('rating').count()['user_id']
    for i, r in enumerate(rating_count):
        print(f"{int(r / l * 100)}% of entries were rated {i+1} stars.")

def train_test_split_df(df, size=0.2, random_state=0):
    s = int(size * len(df))
    df_split = df.sample(n=s, random_state=random_state)
    df_remaining = df.drop(df_split.index)
    return df_remaining, df_split

def create_cbcf_df(cbf_model, cf_model, df):
    cbf_pred = cbf_model.predict(df)
    cf_pred = cf_model.predict(df)
    print(cbf_model.mse, cf_model.mse)
    df['cbf_pred'] = cbf_pred
    df['cf_pred'] = cf_pred

    df = df.drop(columns=GENRES)

    return df


