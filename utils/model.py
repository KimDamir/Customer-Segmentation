from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras import Model


# KMeans model with optional hyperparameter tuning
def build_kmeans_model(df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df)
    return kmeans

# Function to build and compile the deep learning model
def build_deep_learning_model(X_train, encoding_dim):
    input_dim=X_train.shape[1]
    input_tensor = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(input_tensor)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    encoded = Dense(encoding_dim, activation='linear', name='embedding')(x)
    x = Dense(128, activation='relu')(encoded)
    x = Dense(64, activation='relu')(x)
    decoded = Dense(input_dim, activation='linear')(x)

    autoencoder = Model(inputs=input_tensor, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])
    encoder = Model(inputs=input_tensor, outputs=encoded) #Encoder defined directly here.
    return autoencoder, encoder


def perform_hyperparameter_tuning(model, X_train, y_train):
    # Perform hyperparameter tuning with GridSearch
    param_grid = {
        'n_clusters': [2, 3, 4, 5],
        'init': ['k-means++', 'random'],
        'n_init': [10, 20, 30]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_
