"""RetailIQ - Flask API + themed dashboard.

Endpoints
  GET /                      dashboard UI
  GET /api/health            service status
  GET /api/summary           KPI numbers
  GET /api/recommendations   ?status=CRITICAL&store_id=S01&category=Grocery&limit=50
  GET /api/forecast          ?store_id=S01&product_id=P001  (history + forecast)
  GET /api/metrics           model comparison
  GET /api/series            list of store/product combinations
"""
import json
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

from src import config
from src.inventory import summarize

_cache = {}


def _load(path: Path, kind="csv"):
    """Read a file, reloading automatically when it changes on disk."""
    if not path.exists():
        raise FileNotFoundError(path.name)
    mtime = path.stat().st_mtime
    hit = _cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    if kind == "json":
        data = json.loads(path.read_text())
    else:
        data = pd.read_csv(path, parse_dates=["date"]) if "sales" in path.name or "forecast" in path.name else pd.read_csv(path)
    _cache[path] = (mtime, data)
    return data


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.errorhandler(FileNotFoundError)
    def missing(err):
        return jsonify(error=f"{err} not found. Run `python runall.py` first."), 503

    @app.get("/")
    def index():
        return render_template("index.html", theme=config.THEME)

    @app.get("/api/health")
    def health():
        ready = config.RECO_FILE.exists() and config.FORECAST_FILE.exists()
        return jsonify(status="ok", pipeline_outputs_ready=ready, service="RetailIQ API")

    @app.get("/api/summary")
    def summary():
        rec = _load(config.RECO_FILE)
        out = summarize(rec)
        out["best_model"] = _load(config.METRICS_FILE, "json")["best_model"]
        return jsonify(out)

    @app.get("/api/recommendations")
    def recommendations():
        rec = _load(config.RECO_FILE)
        for col in ("status", "store_id", "category", "product_id"):
            val = request.args.get(col)
            if val:
                rec = rec[rec[col].str.upper() == val.upper()]
        limit = request.args.get("limit", type=int)
        if limit:
            rec = rec.head(limit)
        return jsonify(rec.where(rec.notna(), None).to_dict(orient="records"))

    @app.get("/api/forecast")
    def forecast():
        store, prod = request.args.get("store_id"), request.args.get("product_id")
        fc = _load(config.FORECAST_FILE)
        hist = _load(config.CLEAN_SALES_FILE)
        if not store or not prod:
            return jsonify(error="store_id and product_id are required"), 400
        f = fc[(fc.store_id == store) & (fc.product_id == prod)]
        h = hist[(hist.store_id == store) & (hist.product_id == prod)].tail(60)
        if f.empty:
            return jsonify(error="unknown store/product"), 404
        return jsonify(
            store_id=store, product_id=prod,
            history=[{"date": d.strftime("%Y-%m-%d"), "units": int(u)} for d, u in zip(h.date, h.units_sold)],
            forecast=[{"date": d.strftime("%Y-%m-%d"), "units": float(u)} for d, u in zip(f.date, f.forecast)],
        )

    @app.get("/api/metrics")
    def metrics():
        return jsonify(_load(config.METRICS_FILE, "json"))

    @app.get("/api/series")
    def series():
        rec = _load(config.RECO_FILE)
        cols = ["store_id", "city", "product_id", "product_name"]
        return jsonify(rec[cols].sort_values(cols[:1] + cols[2:3]).to_dict(orient="records"))

    return app


if __name__ == "__main__":
    create_app().run(port=5000)
