# evaluate.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import mse
from data.boston_housing_data import load_data

# Load Boston Housing data
(x_train, y_train), (x_test, y_test) = load_data()

# Load the trained model
# model = tf.keras.models.load_model('dnn_boston_housing_model.h5')
model = keras.models.load_model("dnn_boston_housing_model.keras", custom_objects={'mse': mse})


# Evaluate the model
test_loss, test_mae = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest MAE: {test_mae}')
