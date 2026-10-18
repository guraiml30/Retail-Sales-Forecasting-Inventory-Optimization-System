"""Model training, honest multi-day evaluation and future forecasting."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import BASE_COLS, FEATURES, KEYS, TARGET, build_features


def get_models():
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=3,
                                               n_jobs=-1, random_state=config.SEED),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06,
                                                           random_state=config.SEED),
    }


def compute_metrics(actual, predicted):
    err = np.asarray(actual) - np.asarray(predicted)
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "WAPE_%": float(100 * np.abs(err).sum() / max(np.sum(actual), 1e-9))}


def seasonal_naive(history, horizon):
    """Benchmark: 'next week looks like last week'."""
    rows = []
    last = history["date"].max()
    for (s, p), g in history.sort_values("date").groupby(KEYS):
        last7 = g[TARGET].to_numpy()[-7:]
        for i in range(horizon):
            rows.append((last + pd.Timedelta(days=i + 1), s, p, float(last7[i % 7])))
    return pd.DataFrame(rows, columns=["date", "store_id", "product_id", "forecast"])


def recursive_forecast(model, history, horizon):
    """Predict day by day; each prediction becomes the next day's lag feature."""
    last_date = history["date"].max()
    window = history[history["date"] > last_date - pd.Timedelta(days=60)][BASE_COLS].copy()
    combos = window[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    last_price = window.sort_values("date").groupby(KEYS)["price"].last()
    combos["price"] = [last_price[(s, p)] for s, p in zip(combos["store_id"], combos["product_id"])]

    out = []
    for i in range(horizon):
        day = last_date + pd.Timedelta(days=i + 1)
        new = combos.copy()
        new["date"], new["promo"], new[TARGET] = day, 0, np.nan
        window = pd.concat([window, new[BASE_COLS]], ignore_index=True)
        today = build_features(window)
        today = today[today["date"] == day]          # sorted by store, product = combos order
        pred = np.clip(model.predict(today[FEATURES]), 0, None)
        window.loc[window["date"] == day, TARGET] = pred
        out.append(pd.DataFrame({"date": day, "store_id": today["store_id"].to_numpy(),
                                 "product_id": today["product_id"].to_numpy(), "forecast": pred}))
    return pd.concat(out, ignore_index=True)


def run_forecasting(df):
    """Train 3 models, compare on a held-out month, refit best, forecast next 30 days."""
    feats = build_features(df)
    cutoff = df["date"].max() - pd.Timedelta(days=config.HOLDOUT_DAYS)
    train = feats[feats["date"] <= cutoff].dropna(subset=FEATURES + [TARGET])
    history = df[df["date"] <= cutoff]
    actual = df[df["date"] > cutoff][["date", "store_id", "product_id", TARGET]]
    keys = ["date", "store_id", "product_id"]

    results, holdout = {}, {}
    base = actual.merge(seasonal_naive(history, config.HOLDOUT_DAYS), on=keys)
    results["Seasonal Naive (baseline)"] = compute_metrics(base[TARGET], base["forecast"])

    models = get_models()
    for name, model in models.items():
        model.fit(train[FEATURES], train[TARGET])
        merged = actual.merge(recursive_forecast(model, history, config.HOLDOUT_DAYS), on=keys)
        results[name] = compute_metrics(merged[TARGET], merged["forecast"])
        holdout[name] = merged

    best_name = min(models, key=lambda m: results[m]["WAPE_%"])
    best_hold = holdout[best_name].rename(columns={TARGET: "actual", "forecast": "predicted"})
    best_hold.to_csv(config.HOLDOUT_FILE, index=False)

    best_hold["sq"] = (best_hold["actual"] - best_hold["predicted"]) ** 2
    err = best_hold.groupby(KEYS)["sq"].mean().pow(0.5).rename("sigma_error").reset_index()
    err.to_csv(config.ERROR_FILE, index=False)

    # refit on ALL data, then forecast the real future
    final = clone(models[best_name])
    full = feats.dropna(subset=FEATURES + [TARGET])
    final.fit(full[FEATURES], full[TARGET])
    joblib.dump(final, config.MODEL_FILE)
    forecast = recursive_forecast(final, df, config.FORECAST_HORIZON)
    forecast["forecast"] = forecast["forecast"].round(2)
    forecast.to_csv(config.FORECAST_FILE, index=False)

    summary = {"best_model": best_name, "holdout_days": config.HOLDOUT_DAYS,
               "metrics": results,
               "beats_baseline": bool(results[best_name]["WAPE_%"] < results["Seasonal Naive (baseline)"]["WAPE_%"])}
    config.METRICS_FILE.write_text(json.dumps(summary, indent=2))
    return summary, best_hold, forecast, err
