import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.make_windows_tetouan import get_data

# Setting seed for reproducibility
seed = 42

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# Custom layer to divide the input time-series into patches
class PatchLayer(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = inputs.shape[1]
        num_features = inputs.shape[2]

        # Number of patches created from the full input sequence
        num_patches = time_steps // self.patch_size

        # Reshaping input from individual time steps to tokens
        x = tf.reshape(
            inputs,
            (batch_size, num_patches, self.patch_size * num_features)
        )
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size
        })
        return config


# Defining Transformer encoder block
def transformer_encoder(x, embed_dim, num_heads, ff_dim, dropout=0.1):
    
    # Multi-head self-attention
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim
    )(x, x)

    # Dropout for regularization
    attn_output = layers.Dropout(dropout)(attn_output)

    # Residual connection and layer normalization after attention
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    # Feedforward network
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(embed_dim)(ff)
    ff = layers.Dropout(dropout)(ff)

    # Residual connection and layer normalization after feedforward network
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    return x


# Building patch transformer model
def build_patch_transformer(
    input_shape=(168, 5),
    patch_size=7,
    embed_dim=64,
    num_heads=4,
    ff_dim=128,
    num_blocks=2,
    output_window=24,
    dropout=0.1
):
    inputs = layers.Input(shape=input_shape)

    # Extracting patch from input
    x = PatchLayer(patch_size)(inputs)
    
    # Projecting patches onto an embedding space
    x = layers.Dense(embed_dim)(x)

    num_patches = input_shape[0] // patch_size

    # Creating positionl embedding for ordering
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = layers.Embedding(
        input_dim=num_patches,
        output_dim=embed_dim
    )(positions)

    # Adding positional information
    x = x + pos_embed

    # Applying repeated transformer encoder blocks 
    for _ in range(num_blocks):
        x = transformer_encoder(x, embed_dim, num_heads, ff_dim, dropout)

    # Attention pooling
    scores = layers.Dense(1)(x)

    #Converting scores into normalized weights
    weights = layers.Softmax(axis=1)(scores)
    x = layers.Multiply()([x, weights])

    # Summing weighted patches
    x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)

    # Dense forecasting layer 
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    # Final output layer
    outputs = layers.Dense(output_window)(x)

    model = Model(inputs, outputs)

    # Compiling model using Adam optimizer, MSE loss, and MAE metric
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="mse",
        metrics=["mae"]
    )

    return model


#  Saving training and validation loss plots
def plot_loss(history, save_path="figures/patch_transformer_tetouan_loss.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Patch Transformer Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# Saving example 24-hour forecast comparing actual vs predicted values
def plot_forecast(y_true, y_pred, save_path="figures/patch_transformer_tetouan_forecast.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title("Patch Transformer Forecast")
    plt.savefig(save_path)
    plt.close()



def main():

    # Loading Tetouan data using 336-step input window and 24-step output window
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = get_data(
        input_window=336,
        output_window=24
    )

    # Building patch transformer model
    model = build_patch_transformer(
    input_shape=(X_train.shape[1], X_train.shape[2]),
    patch_size=12
    )

    model.summary()

    # Adding early stopping if validation loss stops improving
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # Reducing learning rate when validation loss plateaus
    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # Training the model 
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        callbacks=[early_stop, lr_scheduler]
    )

    # Evaluating model performance on unseen data
    test_mse, test_mae = model.evaluate(X_test, y_test)
    test_rmse = np.sqrt(test_mse)

    print(f"Patch Transformer MSE: {test_mse:.6f}")
    print(f"Patch Transformer RMSE: {test_rmse:.6f}")
    print(f"Patch Transformer MAE: {test_mae:.6f}")

    # Saving training loss plot
    plot_loss(history)

    # Saving example forecast plot
    y_pred = model.predict(X_test[:1])
    plot_forecast(y_test[0], y_pred[0])

    # Saving trained model and weights
    os.makedirs("models", exist_ok=True)
    model.save("models/transformer_tetouan_model.h5")
    model.save_weights("models/transformer_tetouan_weights.weights.h5")


if __name__ == "__main__":
    main()
