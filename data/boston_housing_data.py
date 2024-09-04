# data/boston_housing_data.py

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def load_data():
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()

    # Standardize the features (important for regression tasks)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return (x_train, y_train), (x_test, y_test)
