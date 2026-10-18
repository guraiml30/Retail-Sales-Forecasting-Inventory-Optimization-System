"""Central configuration: paths, simulation settings, business rules, theme."""
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
IMAGES_DIR = ROOT / "images"

RAW_SALES_FILE = DATA_RAW / "retail_sales.csv"
CATALOG_FILE = DATA_RAW / "products.csv"
CLEAN_SALES_FILE = DATA_PROCESSED / "clean_sales.csv"
FORECAST_FILE = OUTPUTS_DIR / "forecast_next30.csv"
HOLDOUT_FILE = OUTPUTS_DIR / "holdout_predictions.csv"
ERROR_FILE = OUTPUTS_DIR / "forecast_error_by_series.csv"
RECO_FILE = OUTPUTS_DIR / "inventory_recommendations.csv"
METRICS_FILE = OUTPUTS_DIR / "metrics.json"
CLEANING_REPORT_FILE = OUTPUTS_DIR / "cleaning_report.json"
MODEL_FILE = MODELS_DIR / "best_model.joblib"


def ensure_dirs():
    for d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR, IMAGES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------- Simulation settings ----------
SEED = 42
END_DATE = "2026-09-30"      # last day of simulated history
N_DAYS = 730                 # two years of daily history
ORIGIN_DATE = "2024-01-01"   # reference date for the trend index
HOLDOUT_DAYS = 30            # days kept aside to test the model
FORECAST_HORIZON = 30        # days to forecast into the future

STORES = [
    {"store_id": "S01", "city": "Delhi", "scale": 1.20},
    {"store_id": "S02", "city": "Mumbai", "scale": 1.00},
    {"store_id": "S03", "city": "Bengaluru", "scale": 0.85},
]

# base_demand = avg units/day per store; festive_sens / summer_sens = seasonal strength
PRODUCTS = [
    {"product_id": "P001", "product_name": "Basmati Rice 5kg", "category": "Grocery", "base_demand": 14, "price": 520, "unit_cost": 410, "lead_time": 5, "festive_sens": 1.3, "summer_sens": 1.0},
    {"product_id": "P002", "product_name": "Sunflower Oil 1L", "category": "Grocery", "base_demand": 18, "price": 165, "unit_cost": 135, "lead_time": 4, "festive_sens": 1.2, "summer_sens": 1.0},
    {"product_id": "P003", "product_name": "Toor Dal 1kg", "category": "Grocery", "base_demand": 12, "price": 160, "unit_cost": 128, "lead_time": 6, "festive_sens": 1.1, "summer_sens": 1.0},
    {"product_id": "P004", "product_name": "Cold Drink 2L", "category": "Beverages", "base_demand": 25, "price": 95, "unit_cost": 70, "lead_time": 3, "festive_sens": 1.0, "summer_sens": 1.6},
    {"product_id": "P005", "product_name": "Packaged Juice 1L", "category": "Beverages", "base_demand": 16, "price": 110, "unit_cost": 80, "lead_time": 3, "festive_sens": 1.0, "summer_sens": 1.4},
    {"product_id": "P006", "product_name": "Tea Powder 500g", "category": "Beverages", "base_demand": 13, "price": 280, "unit_cost": 215, "lead_time": 5, "festive_sens": 1.1, "summer_sens": 0.8},
    {"product_id": "P007", "product_name": "Potato Chips", "category": "Snacks", "base_demand": 30, "price": 20, "unit_cost": 14, "lead_time": 4, "festive_sens": 1.3, "summer_sens": 1.0},
    {"product_id": "P008", "product_name": "Biscuits Family Pack", "category": "Snacks", "base_demand": 22, "price": 60, "unit_cost": 45, "lead_time": 4, "festive_sens": 1.2, "summer_sens": 1.0},
    {"product_id": "P009", "product_name": "Detergent 2kg", "category": "Household", "base_demand": 10, "price": 310, "unit_cost": 240, "lead_time": 7, "festive_sens": 1.1, "summer_sens": 1.0},
    {"product_id": "P010", "product_name": "Dishwash Liquid", "category": "Household", "base_demand": 12, "price": 120, "unit_cost": 88, "lead_time": 6, "festive_sens": 1.0, "summer_sens": 1.0},
    {"product_id": "P011", "product_name": "Face Wash", "category": "Personal Care", "base_demand": 9, "price": 240, "unit_cost": 170, "lead_time": 6, "festive_sens": 1.1, "summer_sens": 1.1},
    {"product_id": "P012", "product_name": "Shampoo 340ml", "category": "Personal Care", "base_demand": 11, "price": 320, "unit_cost": 235, "lead_time": 6, "festive_sens": 1.2, "summer_sens": 1.0},
]

STORE_IDS = [s["store_id"] for s in STORES]
PRODUCT_IDS = [p["product_id"] for p in PRODUCTS]

# ---------- Inventory business rules ----------
ORDER_COST = 300          # INR fixed cost per purchase order
HOLDING_RATE = 0.25       # yearly holding cost as % of unit cost
SERVICE_LEVELS = {"A": 0.98, "B": 0.95, "C": 0.90}
MAX_COVER_DAYS = 21       # never recommend more than ~21 days of stock per order

# ---------- Theme (used by charts AND the web dashboard) ----------
THEME = {
    "bg": "#0B1F2A", "panel": "#12303F", "border": "#1F4557",
    "text": "#E6F1F5", "muted": "#8FB1BF",
    "primary": "#14B8A6", "accent": "#F59E0B",
    "danger": "#EF4444", "ok": "#22C55E", "info": "#38BDF8",
}
