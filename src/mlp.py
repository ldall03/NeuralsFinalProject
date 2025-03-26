from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

def naive_model(df):
    X = df.drop(columns=['rating']).values
    y = df['rating'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mlp = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000, random_state=42)

    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE: ", mse)

    return mlp

def gscv_model(df):
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],  
        'activation': ['relu', 'tanh'],  
        'solver': ['adam', 'svg'],  
        'learning_rate_init': [0.001, 0.01],  
        'max_iter': [500, 1000]  
    }
    X = df.drop(columns=['rating']).values
    y = df['rating'].values

    mlp = MLPRegressor(random_state=42)

    grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)

    model = grid_search.best_estimator_

    return model

def predict(model, u_id, m_id, df):
    u_row = df[df['user_id'] == u_id]
    u_row = u_row[u_row['movie_id'] == m_id]
    if len(u_row) == 0:
        raise Exception("User/Movie pair does not exist in dataset")

    X = u_row.drop(columns=['rating']).values
    y = u_row['rating'].values

    y_pred = model.predict(X)
    return y, y_pred


