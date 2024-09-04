Let's create a TensorFlow project using Keras with the Boston Housing dataset. This dataset contains information about various houses in Boston and is commonly used for regression tasks, where the goal is to predict the price of a house based on various features.

Project Structure
Here's the project structure for the Boston Housing dataset:

```css
tensorflow_boston_housing/
│
├── data/
│   └── boston_housing_data.py  # Script to load Boston Housing data
├── models/
│   ├── dnn_model.py            # Script to define and compile a DNN model
│   ├── cnn_model.py            # Script to define and compile a CNN model (for experimentation)
├── train.py                    # Script to train the model
├── evaluate.py                 # Script to evaluate the trained model
└── utils/
    └── plot_history.py         # Script to plot training history

```

Step 1: Load Boston Housing Data
Create a file named boston_housing_data.py in the data/ directory to load and preprocess the Boston Housing dataset.

```python
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

```

Step 2: Define Models
Below are some models tailored for the Boston Housing dataset:

1. DNN Model for Boston Housing

```python
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
```

2. CNN Model for Boston Housing (Experimental)
While CNNs are typically used for image data, you can experiment with a CNN for tabular data by treating the features as a 1D spatial sequence. This is more of an experimental approach.

```python
# models/cnn_model.py

import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape=(13,)):
    model = models.Sequential([
        layers.Reshape((input_shape[0], 1), input_shape=input_shape),
        layers.Conv1D(32, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])

    model.compile(optimizer='adam',
                  loss='mse',  # Mean Squared Error for regression
                  metrics=['mae'])  # Mean Absolute Error for evaluation
    return model
```

Step 3: Train the Model
Create a train.py script at the root of the project to load data, build the model, and train it.

```python
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
model.save('dnn_boston_housing_model.h5')

# Save the training history
import pickle
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
```


Step 4: Evaluate the Model
Create an evaluate.py script to evaluate the trained model on the test data.

```python
# evaluate.py

import tensorflow as tf
from data.boston_housing_data import load_data

# Load Boston Housing data
(x_train, y_train), (x_test, y_test) = load_data()

# Load the trained model
model = tf.keras.models.load_model('dnn_boston_housing_model.h5')

# Evaluate the model
test_loss, test_mae = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest MAE: {test_mae}')
```

Step 5: Plot Training History
Use the same plot_history.py script to plot the training and validation loss and MAE.

```python
# utils/plot_history.py

import matplotlib.pyplot as plt
import pickle

def plot_history(history_file='history.pkl'):
    with open(history_file, 'rb') as f:
        history = pickle.load(f)

    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend(loc='upper right')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history['mae'], label='MAE')
    plt.plot(history['val_mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    plot_history()
```


Step 6: Run the Project
Train the Model: Run the train.py script to start training the model.

```bash
python train.py
```

Evaluate the Model: After training, evaluate the model's performance using evaluate.py.

```python
python evaluate.py
```

Plot the Training History: Visualize the training history using plot_history.py.

```python
python utils/plot_history.py
```
