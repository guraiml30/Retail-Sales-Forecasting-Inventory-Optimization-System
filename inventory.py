"""Inventory optimisation: safety stock, reorder point, EOQ and recommendations."""
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from src import config
from src.features import KEYS


# ---------- Pure formula functions (easy to unit-test) ----------
def z_score(service_level):
    return NormalDist().inv_cdf(service_level)


def safety_stock(sigma_daily, lead_time, service_level):
    """SS = z x sigma x sqrt(lead time)."""
    return math.ceil(z_score(service_level) * sigma_daily * math.sqrt(lead_time))


def reorder_point(avg_daily_demand, lead_time, ss):
    """ROP = demand during lead time + safety stock."""
    return math.ceil(avg_daily_demand * lead_time + ss)


def eoq(annual_demand, order_cost, holding_cost_per_unit):
    """Economic Order Quantity = sqrt(2DS / H)."""
    if holding_cost_per_unit <= 0:
        return 0.0
    return math.sqrt(2 * annual_demand * order_cost / holding_cost_per_unit)


def classify_status(stock, avg_daily, lead_time, rop, max_stock):
    if stock < avg_daily * lead_time:
        return "CRITICAL"      # will run out before a new order can arrive
    if stock <= rop:
        return "REORDER"
    if stock > max_stock or stock / avg_daily > config.MAX_COVER_DAYS:
        return "OVERSTOCK"
    return "HEALTHY"


def abc_classes(df, days=90):
    """ABC analysis on recent revenue: A = top 70%, B = next 20%, C = rest."""
    recent = df[df["date"] > df["date"].max() - pd.Timedelta(days=days)].copy()
    recent["revenue"] = recent["units_sold"] * recent["price"]
    rev = recent.groupby(KEYS)["revenue"].sum().sort_values(ascending=False)
    cum = rev.cumsum() / rev.sum()
    cls = np.where(cum <= 0.70, "A", np.where(cum <= 0.90, "B", "C"))
    cls[0] = "A"
    return pd.Series(cls, index=rev.index, name="abc_class").reset_index()


# ---------- Main routine ----------
def build_recommendations(clean, forecast, errors, catalog):
    last_day = clean["date"].max()
    stock = (clean[clean["date"] == last_day][KEYS + ["city", "stock_on_hand"]]
             .rename(columns={"stock_on_hand": "current_stock"}))
    demand = (forecast.groupby(KEYS)["forecast"].agg(["mean", "sum"]).reset_index()
              .rename(columns={"mean": "avg_daily_forecast", "sum": "forecast_30d_units"}))

    df = (stock.merge(demand, on=KEYS).merge(errors, on=KEYS)
          .merge(catalog, on="product_id").merge(abc_classes(clean), on=KEYS))
    df["service_level"] = df["abc_class"].map(config.SERVICE_LEVELS)

    rows = []
    for r in df.itertuples(index=False):
        d = max(r.avg_daily_forecast, 0.01)
        ss = safety_stock(max(r.sigma_error, 0.5), r.lead_time, r.service_level)
        rop = reorder_point(d, r.lead_time, ss)
        holding = config.HOLDING_RATE * r.unit_cost
        q = eoq(d * 365, config.ORDER_COST, holding)
        q = min(max(q, d * 7), d * config.MAX_COVER_DAYS)       # keep it practical
        max_stock = rop + q
        status = classify_status(r.current_stock, d, r.lead_time, rop, max_stock)
        order_qty = math.ceil(max_stock - r.current_stock) if status in ("CRITICAL", "REORDER") else 0
        shortfall = max(0.0, d * r.lead_time - r.current_stock)
        excess = max(0.0, r.current_stock - max_stock)
        rows.append({
            **{k: getattr(r, k) for k in ("store_id", "city", "product_id", "product_name", "category", "abc_class")},
            "current_stock": int(r.current_stock), "avg_daily_forecast": round(d, 2),
            "forecast_30d_units": round(r.forecast_30d_units, 1), "lead_time_days": int(r.lead_time),
            "service_level": r.service_level, "safety_stock": ss, "reorder_point": rop,
            "eoq": round(q), "max_stock": round(max_stock),
            "days_of_cover": round(r.current_stock / d, 1), "status": status,
            "recommended_order_qty": int(order_qty),
            "order_value": round(order_qty * r.unit_cost, 2),
            "revenue_at_risk": round(shortfall * r.price, 2),
            "excess_value": round(excess * r.unit_cost, 2),
        })
    out = pd.DataFrame(rows)
    out["action"] = out.apply(_action_text, axis=1)
    order = {"CRITICAL": 0, "REORDER": 1, "OVERSTOCK": 2, "HEALTHY": 3}
    out = out.sort_values(by="status", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)
    out.to_csv(config.RECO_FILE, index=False)
    return out


def _action_text(r):
    if r["status"] == "CRITICAL":
        return f"URGENT: order {r['recommended_order_qty']} units now - stock covers only {r['days_of_cover']} days (lead time {r['lead_time_days']}d)"
    if r["status"] == "REORDER":
        return f"Place order for {r['recommended_order_qty']} units - stock is at/below reorder point {r['reorder_point']}"
    if r["status"] == "OVERSTOCK":
        return f"Hold orders / consider promotion - {r['days_of_cover']} days of cover (max {r['max_stock']} units)"
    return "No action - stock level is healthy"


def summarize(rec):
    return {
        "total_skus": int(len(rec)),
        "status_counts": {k: int(v) for k, v in rec["status"].value_counts().items()},
        "total_order_value": round(float(rec["order_value"].sum()), 2),
        "revenue_at_risk": round(float(rec["revenue_at_risk"].sum()), 2),
        "excess_stock_value": round(float(rec["excess_value"].sum()), 2),
        "avg_days_of_cover": round(float(rec["days_of_cover"].mean()), 1),
    }
