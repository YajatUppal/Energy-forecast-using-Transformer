import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_tetouan_data():
    # Loading raw time-series data
    df = pd.read_csv("data/powerconsumption.csv")

    # Converting datetime column
    df["Datetime"] = pd.to_datetime(df["Datetime"])


    # Creating Cyclical Time features
    df["hour"] = df["Datetime"].dt.hour
    df["dayofweek"] = df["Datetime"].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    feature_cols = [
        "Temperature",
        "Humidity",
        "WindSpeed",
        "GeneralDiffuseFlows",
        "DiffuseFlows",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos"
    ]

    target_col = "PowerConsumption_Zone1"

    features = df[feature_cols].values
    target = df[target_col].values.reshape(-1, 1)

    # Scaling features and target
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    features = feature_scaler.fit_transform(features)
    target = target_scaler.fit_transform(target)

    return features, target, target_scaler