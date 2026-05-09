import numpy as np
from src.preprocessing_tetouan import load_tetouan_data

# Converting continuous time-series into samples for supervized learning 
def create_windows(features, target, input_window=336, output_window=24):
    X = []
    y = []

    # Going through the time-series to create input-output pairs
    for i in range(len(features) - input_window - output_window):
        X.append(features[i:i+input_window])

        y.append(
            target[
                i+input_window:i+input_window+output_window
            ].flatten()
        )

    return np.array(X), np.array(y)


def get_data(input_window=336, output_window=24):
    
    # Loading and preprocessing
    features, target, scaler = load_tetouan_data()

    # Converting data into forecasting windows
    X, y = create_windows(
        features,
        target,
        input_window,
        output_window
    )
    # Computing split sizes
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]

    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler
    )