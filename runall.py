"""One command to run the entire pipeline and launch the API.

    python runall.py                 # full pipeline + start API on http://127.0.0.1:5000
    python runall.py --no-api        # pipeline only
    python runall.py --skip-data     # reuse existing data/raw/retail_sales.csv
    python runall.py --port 8000
"""
import argparse
import time

from src import config
from src.data_generator import save_raw_data
from src.forecasting import run_forecasting
from src.inventory import build_recommendations, summarize
from src.preprocessing import load_raw, run_cleaning
from src.visualization import plot_eda, plot_forecast, plot_inventory
import pandas as pd


def step(n, total, title):
    print(f"\n[{n}/{total}] {title} ...", flush=True)
    return time.time()


def done(t0):
    print(f"      done in {time.time() - t0:.1f}s", flush=True)


def run_pipeline(skip_data=False):
    config.ensure_dirs()
    total = 6

    t = step(1, total, "Generating simulated retail data" if not skip_data else "Using existing raw data")
    if not skip_data or not config.RAW_SALES_FILE.exists():
        raw = save_raw_data()
    else:
        raw = load_raw()
    print(f"      raw rows: {len(raw):,}  ->  {config.RAW_SALES_FILE.name}")
    done(t)

    t = step(2, total, "Cleaning & validating data")
    clean, report = run_cleaning()
    print(f"      duplicates removed: {report['duplicates_removed']} | missing before: {report['missing_before']} | after: {report['missing_after'] or 'none'}")
    print(f"      series: {report['series']} x days: {report['days']} | complete grid: {report['grid_complete']}")
    done(t)

    t = step(3, total, "Exploratory analysis charts")
    plot_eda(clean)
    done(t)

    t = step(4, total, "Training models & forecasting next 30 days")
    summary, holdout, forecast, errors = run_forecasting(clean)
    for name, m in summary["metrics"].items():
        flag = "  <-- best" if name == summary["best_model"] else ""
        print(f"      {name:<28} MAE {m['MAE']:.2f} | RMSE {m['RMSE']:.2f} | WAPE {m['WAPE_%']:.1f}%{flag}")
    plot_forecast(clean, summary, holdout, forecast)
    done(t)

    t = step(5, total, "Optimising inventory (safety stock, ROP, EOQ)")
    catalog = pd.read_csv(config.CATALOG_FILE)
    rec = build_recommendations(clean, forecast, errors, catalog)
    kpi = summarize(rec)
    print(f"      status counts: {kpi['status_counts']}")
    print(f"      suggested order value: INR {kpi['total_order_value']:,.0f} | revenue at risk: INR {kpi['revenue_at_risk']:,.0f}")
    done(t)

    t = step(6, total, "Saving inventory charts")
    plot_inventory(rec)
    done(t)
    print("\nPipeline complete. Outputs in /outputs, charts in /images, model in /models.")


def main():
    ap = argparse.ArgumentParser(description="Run the full Retail Forecasting pipeline and API")
    ap.add_argument("--no-api", action="store_true", help="skip starting the web API")
    ap.add_argument("--skip-data", action="store_true", help="reuse existing raw data")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    run_pipeline(skip_data=args.skip_data)

    if not args.no_api:
        from app.api import create_app
        print(f"\nStarting RetailIQ dashboard + API at http://{args.host}:{args.port}  (Ctrl+C to stop)")
        create_app().run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
