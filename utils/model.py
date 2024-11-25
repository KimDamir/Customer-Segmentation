from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# KMeans model with optional hyperparameter tuning
def build_kmeans_model(df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df)
    return kmeans

# Function to build and compile the deep learning model
def build_deep_learning_model(X_train):
    model = Sequential()
    
    # Input layer
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    
    # Hidden layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))  # Regularization
    
    model.add(Dense(64, activation='relu'))
    
    # Output layer (for 3 classes: Tier 1, Tier 2, Tier 3)
    model.add(Dense(3, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model


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
