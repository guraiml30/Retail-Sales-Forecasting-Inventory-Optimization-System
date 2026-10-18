"""Data loading and cleaning."""
import json
import pandas as pd
from src import config
from src.features import KEYS


def load_raw(path=config.RAW_SALES_FILE):
    return pd.read_csv(path, parse_dates=["date"])


def clean_sales(df):
    """Remove duplicates, fix missing prices, clip negatives, validate grid."""
    report = {"rows_raw": int(len(df)),
              "missing_before": {k: int(v) for k, v in df.isna().sum().items() if v > 0}}

    dup_mask = df.duplicated(subset=["date"] + KEYS)
    report["duplicates_removed"] = int(dup_mask.sum())
    df = df[~dup_mask].sort_values(KEYS + ["date"]).reset_index(drop=True)

    df["price"] = df.groupby(KEYS)["price"].transform(lambda s: s.interpolate(limit_direction="both"))
    report["negative_sales_fixed"] = int((df["units_sold"] < 0).sum())
    df["units_sold"] = df["units_sold"].clip(lower=0)

    n_series = df.groupby(KEYS).ngroups
    n_days = df["date"].nunique()
    report["series"] = int(n_series)
    report["days"] = int(n_days)
    report["grid_complete"] = bool(len(df) == n_series * n_days)
    report["missing_after"] = {k: int(v) for k, v in df.isna().sum().items() if v > 0}
    report["rows_clean"] = int(len(df))
    return df, report


def run_cleaning():
    raw = load_raw()
    clean, report = clean_sales(raw)
    clean.to_csv(config.CLEAN_SALES_FILE, index=False)
    config.CLEANING_REPORT_FILE.write_text(json.dumps(report, indent=2))
    return clean, report
