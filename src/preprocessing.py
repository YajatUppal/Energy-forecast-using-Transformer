import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(data_path="data/PJME_hourly.csv"):
    # Loading raw time-series data
    df = pd.read_csv(data_path)

    # Converting datetime column
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime")
    df = df.set_index("Datetime")

    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep="first")]

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")

    df = df.reindex(full_range)

    # Interpolating missing values
    df["PJME_MW"] = df["PJME_MW"].interpolate(method="linear")

    # Scaling energy consumption
    scaler = MinMaxScaler()
    df["PJME_MW"] = scaler.fit_transform(df[["PJME_MW"]])

    # Creating Cyclical Time features
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    features = [
        "PJME_MW",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    return df[features], scaler