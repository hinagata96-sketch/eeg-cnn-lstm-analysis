import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def prepare_cnn_lstm_data(eeg_files, selected_indices, desired_timesteps=90000):
    """Return: X (samples, timesteps, features), y (labels), class_names"""
    data = []
    labels = []
    class_names = list(eeg_files.keys())
    for class_idx, class_name in enumerate(class_names):
        for csv_path in eeg_files[class_name]:
            df = pd.read_csv(csv_path)
            arr = df.values.astype(np.float32)[:, selected_indices]
            # Pad or truncate to fixed timesteps
            if arr.shape[0] < desired_timesteps:
                pad_width = desired_timesteps - arr.shape[0]
                arr = np.pad(arr, ((0, pad_width), (0, 0)), mode='constant')
            elif arr.shape[0] > desired_timesteps:
                arr = arr[:desired_timesteps, :]
            data.append(arr)
            labels.append(class_idx)
    X = np.stack(data)  # shape: (samples, timesteps, features)
    y = np.array(labels)
    return X, y, class_names

def build_cnn_lstm_model(input_shape, num_classes, cnn_config=None, lstm_units=64, dense_units=64, dropout_rate=0.3):
    """Build a simple CNN-LSTM model."""
    if cnn_config is None:
        cnn_config = [
            {'filters': 32, 'kernel_size': 5, 'pool_size': 2},
            {'filters': 64, 'kernel_size': 5, 'pool_size': 2}
        ]
    model = Sequential()
    # First CNN layer
    first = cnn_config[0]
    model.add(Conv1D(filters=first['filters'], kernel_size=first['kernel_size'], activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=first['pool_size']))
    model.add(Dropout(dropout_rate))
    # Additional CNN layers
    for layer in cnn_config[1:]:
        model.add(Conv1D(filters=layer['filters'], kernel_size=layer['kernel_size'], activation='relu'))
        model.add(MaxPooling1D(pool_size=layer['pool_size']))
        model.add(Dropout(dropout_rate))
    # LSTM and Dense
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# Example usage after ICA and channel selection
def main_cnn_lstm(eeg_files, selected_indices, desired_timesteps=90000, epochs=10, batch_size=4):
    X, y, class_names = prepare_cnn_lstm_data(eeg_files, selected_indices, desired_timesteps)
    num_classes = len(class_names)
    y_cat = to_categorical(y, num_classes)
    model = build_cnn_lstm_model(
        input_shape=(X.shape[1], X.shape[2]),
        num_classes=num_classes
    )
    history = model.fit(X, y_cat, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return model, history, X, y_cat, class_names

# To use, after your ICA/channel selection steps:
# model, history, X, y_cat, class_names = main_cnn_lstm(eeg_files, selected_indices)