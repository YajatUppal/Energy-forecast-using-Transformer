import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from src.make_windows import get_data

# Setting seed for reproducibility
SEED  = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_lstm_model(input_shape=(168, 1), output_window=24):
    model = Sequential([

        # 32 hidden units
        LSTM(32, input_shape=input_shape),
        # Dense output layer
        Dense(output_window)
    ])

    # Compile model using Adam, MSE and MAE
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model

# Saving training and validation loss curve
def plot_loss(history, save_path="figures/lstm_loss_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("LSTM Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# Saving example 24-hour forecast comparing actual vs predicted values
def plot_forecast(y_true, y_pred, save_path="figures/lstm_forecast_example.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title("LSTM 24-Hour Forecast: Actual vs Predicted")
    plt.xlabel("Hour Ahead")
    plt.ylabel("Normalized Energy Consumption")
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = get_data()

    # Building LSTM model
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_window=y_train.shape[1]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    # Train LSTM model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=[early_stop]
    )

    # Evaluate model performance
    test_mse, test_mae = model.evaluate(X_test, y_test)
    test_rmse = np.sqrt(test_mse)

    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")

    # Save training loss plot
    plot_loss(history)

    # Save example forecast plot
    y_pred = model.predict(X_test[:1])
    plot_forecast(y_test[0], y_pred[0])

    # Save trained model 
    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_model.h5")


if __name__ == "__main__":
    main()
