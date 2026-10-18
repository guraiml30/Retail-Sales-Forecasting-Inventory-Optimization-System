"""Virtual retail simulation: creates realistic daily sales + stock levels."""
import numpy as np
import pandas as pd
from src import config
from src.features import festive_flag


def build_catalog():
    cols = ["product_id", "product_name", "category", "price", "unit_cost", "lead_time"]
    return pd.DataFrame(config.PRODUCTS)[cols]


def generate_sales(seed=config.SEED):
    """Simulate 2 years x 3 stores x 12 products of daily sales and stock."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=config.END_DATE, periods=config.N_DAYS, freq="D")
    n = len(dates)
    t = np.arange(n)
    weekend = (dates.dayofweek.to_numpy() >= 5).astype(float)
    summer = np.isin(dates.month.to_numpy(), [4, 5, 6]).astype(float)
    festive = festive_flag(dates)

    frames = []
    for store in config.STORES:
        for p in config.PRODUCTS:
            base = p["base_demand"] * store["scale"] * rng.uniform(0.9, 1.1)

            # ---- demand = base x trend x weekly x festive x summer x promo x noise
            trend = 1 + 0.12 * t / n
            weekly = 1 + 0.20 * weekend
            fest = 1 + (p["festive_sens"] - 1) * 2.0 * festive
            summ = 1 + (p["summer_sens"] - 1) * summer
            starts = rng.random(n) < 0.025                       # promo starts
            promo = (np.convolve(starts.astype(int), np.ones(5, dtype=int))[:n] > 0).astype(int)
            lam = base * trend * weekly * fest * summ * (1 + 0.35 * promo) * rng.lognormal(0, 0.12, n)
            demand = rng.poisson(lam)
            price = p["price"] * (1 + 0.05 * t / n) * (1 - 0.15 * promo)

            # ---- stock simulation with a simple (imperfect) reorder policy
            lead = p["lead_time"]
            # most SKUs are managed sensibly; ~25% are over-ordered, ~15% are under-managed
            style = rng.choice(["ok", "over", "lean"], p=[0.60, 0.25, 0.15])
            rp_mult = {"ok": rng.uniform(1.6, 2.4), "over": rng.uniform(2.0, 3.0), "lean": rng.uniform(0.9, 1.3)}[style]
            qty_mult = {"ok": rng.uniform(9, 16), "over": rng.uniform(28, 40), "lean": rng.uniform(6, 9)}[style]
            reorder_point = base * lead * rp_mult
            order_qty = int(round(base * qty_mult))
            stock = int(base * rng.uniform(12, 25))
            arrivals, pending = {}, False
            sold, eod, receipts, stockout = (np.zeros(n, dtype=int) for _ in range(4))
            for i in range(n):
                if i in arrivals:
                    qty = arrivals.pop(i)
                    stock += qty
                    receipts[i] = qty
                    pending = False
                s = min(int(demand[i]), stock)
                stock -= s
                sold[i] = s
                stockout[i] = int(s < demand[i])
                season = fest[i] * summ[i] * trend[i]          # planners adjust for the season
                if stock <= reorder_point * season and not pending:
                    arrivals[i + lead + int(rng.integers(0, 3))] = int(round(order_qty * season))
                    pending = True
                eod[i] = stock

            frames.append(pd.DataFrame({
                "date": dates, "store_id": store["store_id"], "city": store["city"],
                "product_id": p["product_id"], "product_name": p["product_name"],
                "category": p["category"], "price": price.round(2), "promo": promo,
                "units_sold": sold, "stock_on_hand": eod,
                "stock_received": receipts, "stockout_flag": stockout,
            }))

    df = pd.concat(frames, ignore_index=True)

    # ---- inject realistic "dirty data" so the cleaning step has real work to do
    miss_idx = rng.choice(len(df), size=int(0.003 * len(df)), replace=False)
    df.loc[miss_idx, "price"] = np.nan
    dupes = df.sample(20, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True).sort_values(["date", "store_id", "product_id"])
    return df.reset_index(drop=True)


def save_raw_data(seed=config.SEED):
    config.ensure_dirs()
    df = generate_sales(seed)
    df.to_csv(config.RAW_SALES_FILE, index=False)
    build_catalog().to_csv(config.CATALOG_FILE, index=False)
    return df
