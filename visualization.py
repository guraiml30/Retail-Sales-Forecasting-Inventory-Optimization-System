"""All charts. Saved as PNG into /images so they can go straight into the README."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src import config

T = config.THEME


def _style():
    plt.rcParams.update({
        "figure.facecolor": T["bg"], "axes.facecolor": T["panel"], "axes.edgecolor": T["border"],
        "axes.labelcolor": T["text"], "xtick.color": T["muted"], "ytick.color": T["muted"],
        "text.color": T["text"], "axes.titlecolor": T["text"], "axes.grid": True,
        "grid.color": T["border"], "grid.alpha": 0.5, "legend.facecolor": T["panel"],
        "legend.edgecolor": T["border"], "font.size": 10, "axes.titlesize": 13,
    })


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(config.IMAGES_DIR / name, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_eda(clean):
    _style()
    clean = clean.assign(revenue=clean["units_sold"] * clean["price"])

    daily = clean.groupby("date")["units_sold"].sum()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(daily.index, daily.values, color=T["muted"], alpha=0.35, lw=0.8, label="Daily units")
    ax.plot(daily.index, daily.rolling(7).mean(), color=T["primary"], lw=2, label="7-day average")
    ax.set(title="Total Daily Sales Trend (all stores, all products)", ylabel="Units sold")
    ax.legend()
    _save(fig, "01_sales_trend.png")

    cat = clean.groupby("category")["revenue"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(cat.index, cat.values / 1e6, color=T["primary"])
    ax.set(title="Revenue by Category", xlabel="Revenue (INR millions)")
    _save(fig, "02_category_sales.png")

    heat = clean.assign(month=clean["date"].dt.month).pivot_table(
        index="category", columns="month", values="units_sold", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(heat, cmap="YlGnBu", ax=ax, cbar_kws={"label": "Avg units / day"}, linewidths=0.5, linecolor=T["bg"])
    ax.set(title="Seasonality Heatmap (category x month)", xlabel="Month", ylabel="")
    ax.grid(False)
    _save(fig, "03_seasonality_heatmap.png")

    dow = clean.assign(dow=clean["date"].dt.day_name()).groupby("dow")["units_sold"].mean()
    dow = dow.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dow.index.str[:3], dow.values, color=[T["primary"]] * 5 + [T["accent"]] * 2)
    ax.set(title="Average Units Sold by Day of Week", ylabel="Units / series / day")
    _save(fig, "04_weekday_pattern.png")


def plot_forecast(clean, summary, holdout, forecast):
    _style()
    names = list(summary["metrics"])
    wape = [summary["metrics"][n]["WAPE_%"] for n in names]
    colors = [T["accent"] if n == summary["best_model"] else T["border"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, wape, color=colors)
    ax.set(title="Model Comparison on 30-Day Holdout (lower is better)", ylabel="WAPE %")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save(fig, "05_model_comparison.png")

    agg = holdout.groupby("date")[["actual", "predicted"]].sum()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(agg.index, agg["actual"], color=T["info"], lw=2, label="Actual")
    ax.plot(agg.index, agg["predicted"], color=T["accent"], lw=2, ls="--", label="Predicted")
    ax.set(title=f"Actual vs Predicted - {summary['best_model']} (30-day holdout)", ylabel="Total units")
    ax.legend()
    _save(fig, "06_actual_vs_predicted.png")

    hist = clean[clean["date"] > clean["date"].max() - __import__("pandas").Timedelta(days=90)]
    h = hist.groupby("date")["units_sold"].sum()
    f = forecast.groupby("date")["forecast"].sum()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(h.index, h.values, color=T["info"], lw=2, label="History (last 90 days)")
    ax.plot(f.index, f.values, color=T["primary"], lw=2.5, label="Forecast (next 30 days)")
    ax.axvline(h.index.max(), color=T["muted"], ls=":")
    ax.set(title="Demand Forecast - Next 30 Days", ylabel="Total units")
    ax.legend()
    _save(fig, "07_forecast_next30.png")


def plot_inventory(rec):
    _style()
    colors = {"CRITICAL": T["danger"], "REORDER": T["accent"], "OVERSTOCK": T["info"], "HEALTHY": T["ok"]}
    counts = rec["status"].value_counts().reindex(list(colors)).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color=[colors[s] for s in counts.index])
    ax.set(title="Inventory Health (store x product)", ylabel="Number of SKUs")
    _save(fig, "08_inventory_status.png")

    top = rec[rec["order_value"] > 0].nlargest(10, "order_value").iloc[::-1]
    if len(top):
        labels = top["store_id"] + " - " + top["product_name"]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(labels, top["order_value"], color=[colors[s] for s in top["status"]])
        ax.set(title="Top Replenishment Orders by Value", xlabel="Order value (INR)")
        _save(fig, "09_top_reorders.png")
