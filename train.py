# train.py

import tensorflow as tf
from data.boston_housing_data import load_data
from models.dnn_model import build_dnn_model  # or import the CNN model for experimentation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load Boston Housing data
(x_train, y_train), (x_test, y_test) = load_data()

# Build the DNN model (or the CNN model)
model = build_dnn_model()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

# Train the model
history = model.fit(x_train, y_train, epochs=100, 
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping, reduce_lr])

# Save the trained model
model.save('dnn_boston_housing_model.keras')

# Save the training history
import pickle
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
