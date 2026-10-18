"""Feature engineering: calendar, lag and rolling features."""
import numpy as np
import pandas as pd
from src import config

KEYS = ["store_id", "product_id"]
TARGET = "units_sold"
BASE_COLS = ["date", "store_id", "product_id", "price", "promo", "units_sold"]

FEATURES = [
    "store_code", "product_code", "price", "promo",
    "dow", "month", "is_weekend", "is_festive", "doy_sin", "doy_cos", "t_index",
    "lag_1", "lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28",
]


def festive_flag(dates):
    """1.0 in festive season (Oct 1 - Nov 15), 0.8 around New Year, else 0."""
    d = pd.DatetimeIndex(dates)
    m, day = d.month.to_numpy(), d.day.to_numpy()
    flag = ((m == 10) | ((m == 11) & (day <= 15))).astype(float)
    newyear = ((m == 12) & (day >= 20)) | ((m == 1) & (day <= 2))
    return np.where(newyear, 0.8, flag)


def build_features(df):
    """Return a sorted copy of df with all model features added."""
    df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
    g = df.groupby(KEYS)[TARGET]

    for lag in (1, 7, 14, 28):
        df[f"lag_{lag}"] = g.shift(lag)
    for w in (7, 28):  # shift(1) so today's sales never leak into its own feature
        df[f"roll_mean_{w}"] = g.transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())

    dt = df["date"].dt
    df["dow"] = dt.dayofweek
    df["month"] = dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_festive"] = festive_flag(df["date"])
    doy = dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["t_index"] = (df["date"] - pd.Timestamp(config.ORIGIN_DATE)).dt.days
    df["store_code"] = df["store_id"].map({s: i for i, s in enumerate(config.STORE_IDS)}).astype(int)
    df["product_code"] = df["product_id"].map({p: i for i, p in enumerate(config.PRODUCT_IDS)}).astype(int)
    return df
