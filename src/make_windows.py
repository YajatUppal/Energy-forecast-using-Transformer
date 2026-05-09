import numpy as np
from src.preprocessing import load_and_preprocess_data

# Converting continuous time-series into samples for supervized learning 
def create_windows(data, input_window=168, output_window=24):
    X = []
    y = []

    # Going through the time-series to create input-output pairs
    for i in range(len(data) - input_window - output_window):
        X.append(data[i : i + input_window, :])
        y.append(data[i + input_window : i + input_window + output_window, 0])

    return np.array(X), np.array(y)


def train_val_test_split(X, y, train_ratio=0.70, val_ratio=0.15):

    # Computing split sizes
    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size : train_size + val_size]
    y_val = y[train_size : train_size + val_size]

    X_test = X[train_size + val_size :]
    y_test = y[train_size + val_size :]

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_data(input_window=168, output_window=24):

    # Loading and preprocessing
    df, scaler = load_and_preprocess_data()

    # Selecting input features
    data = df[
        [
            "PJME_MW",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        ]
    ].values

    # Converting data into forecasting windows
    X, y = create_windows(data, input_window, output_window)

    # Splitting windows chronologically
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler