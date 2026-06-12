# model_training.py
import os
import logging

# Mute system, library logging, and oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def generate_trend_sequences(samples_per_class=600, length=50):
    """Generates synthetic trend waveforms (Up-Trend, Down-Trend, Cyclic)."""
    X = []
    y = []
    t = np.linspace(0, 4, length)
    
    for _ in range(samples_per_class):
        # Class 0: Up-Trend (Linear increase with noise)
        noise = np.random.normal(0, 0.15, length)
        up_trend = 0.4 * t + noise
        X.append(up_trend)
        y.append(0)
        
        # Class 1: Down-Trend (Linear decrease with noise)
        noise = np.random.normal(0, 0.15, length)
        down_trend = -0.4 * t + noise
        X.append(down_trend)
        y.append(1)
        
        # Class 2: Cyclic/Range (Oscillating pattern with noise)
        noise = np.random.normal(0, 0.15, length)
        cyclic = 0.8 * np.sin(3.14 * t) + noise
        X.append(cyclic)
        y.append(2)
        
    X = np.array(X)
    y = np.array(y)
    
    # Reshape to match sequential inputs (samples, timesteps, features)
    X = np.expand_dims(X, axis=-1)
    return X, y

def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Constructs a single standard Transformer Encoder block."""
    # Normalization and Multi-Head Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed-Forward Network
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_transformer(input_shape=(50, 1), head_size=64, num_heads=2, ff_dim=64, num_blocks=1, dropout=0.1):
    """Assembles a vanilla Transformer sequence classifier."""
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Stack Transformer blocks
    for _ in range(num_blocks):
        x = transformer_encoder_block(x, head_size, num_heads, ff_dim, dropout)
        
    # Global average pooling over the temporal dimension
    x = layers.GlobalAveragePooling1D()(x)
    
    # Fully connected head
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    length = 50
    X, y = generate_trend_sequences(samples_per_class=800, length=length)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    y_train_cat = to_categorical(y_train, 3)
    y_val_cat = to_categorical(y_val, 3)
    
    model = build_transformer(input_shape=(length, 1))
    model.summary()
    
    print("\nTraining Transformer Sequence Classifier...")
    model.fit(
        X_train, y_train_cat,
        epochs=15,
        batch_size=32,
        validation_data=(X_val, y_val_cat),
        verbose=1
    )
    
    # Save using standard .keras format
    model_name = "transformer_model.keras"
    model.save(model_name)
    print(f"\nModel training finished. Saved model artifact as '{model_name}'.")

if __name__ == '__main__':
    main()