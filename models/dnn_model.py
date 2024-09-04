# models/dnn_model.py

import tensorflow as tf
from tensorflow.keras import layers, models

def build_dnn_model(input_shape=(13,)):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])

    model.compile(optimizer='adam',
                  loss='mse',  # Mean Squared Error for regression
                  metrics=['mae'])  # Mean Absolute Error for evaluation
    return model
