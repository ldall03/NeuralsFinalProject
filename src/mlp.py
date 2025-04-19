from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

import numpy as np

def gscv_model(df):
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],  
        'activation': ['relu', 'tanh'],  
        'solver': ['adam', 'sgd'],  
        'learning_rate_init': [0.001, 0.01],  
        'max_iter': [100, 300]  
    }

    X_train, X_test, y_train, y_test = _prepare_data(df)

    mlp = MLPRegressor(random_state=42)
    grid_search = GridSearchCV(mlp, param_grid, cv=7, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    return model

def classifier_model(df):
    X_train, X_test, y_train, y_test = _prepare_data(df)

    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='tanh', solver='adam', max_iter=300, random_state=42)

    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    mse = accuracy_score(y_test, y_pred)
    print("Acc: ", mse)

    return mlp

def naive_model(df):
    X_train, X_test, y_train, y_test = _prepare_data(df)

    mlp = MLPRegressor(hidden_layer_sizes=(100,), activation='tanh', solver='adam', max_iter=300, random_state=42)

    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE: ", mse)

    return mlp

def undersampled_model(df):
    X_train, X_test, y_train, y_test = _prepare_data(df, scaled=False)

    undersampler = RandomUnderSampler()
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)

    mlp = MLPRegressor(hidden_layer_sizes=(100,), activation='tanh', solver='adam', max_iter=300, random_state=42)

    mlp.fit(X_train_scaled, y_resampled)

    y_pred = mlp.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE: ", mse)

    return mlp

def oversampled_model(df):
    X_train, X_test, y_train, y_test = _prepare_data(df, scaled=False)

    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)

    mlp = MLPRegressor(hidden_layer_sizes=(100,), activation='tanh', solver='adam', max_iter=300, random_state=42)

    mlp.fit(X_train_scaled, y_resampled)

    y_pred = mlp.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE: ", mse)

    return mlp

def log_trans_model(df):
    X_train, X_test, y_train, y_test = _prepare_data(df)
    y_train_log = np.log1p(y_train)

    mlp = MLPRegressor(hidden_layer_sizes=(100,), activation='tanh', solver='adam', max_iter=300, random_state=42)
    mlp.fit(X_train, y_train_log)

    y_pred_log = mlp.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE: ", mse)

    return mlp

def cbcf_mlp(df):
    y = df['rating'].values
    X = df.drop(columns=['rating']).values

    # mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', solver='adam', max_iter=300, random_state=42)
    # mlp.fit(X, y)

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],  
        'activation': ['relu', 'tanh'],  
        'solver': ['adam', 'sgd'],  
        'learning_rate_init': [0.001, 0.01],  
        'max_iter': [100, 300]  
    }

    mlp = MLPClassifier(random_state=42)
    grid_search = GridSearchCV(mlp, param_grid, cv=7, n_jobs=-1)
    grid_search.fit(X, y)

    model = grid_search.best_estimator_
    print(grid_search.best_params_)

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

def _prepare_data(df, scaled=True):
    X = df.drop(columns=['rating']).values
    y = df['rating'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if not scaled:
        return X_train, X_test, y_train, y_test

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

