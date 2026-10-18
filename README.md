# RetailIQ - Retail Sales Forecasting & Inventory Optimization System

> End-to-end retail analytics project: simulated sales data -> cleaning -> EDA -> ML demand forecasting -> safety stock / reorder point / EOQ -> themed web dashboard + REST API. One command runs everything.

![Dashboard](images/dashboard_home.png)

## Problem Statement
Retailers lose money in two ways: **stockouts** (lost sales) and **overstock** (blocked cash, wastage). This project forecasts demand for every store-product pair and converts forecasts into concrete replenishment decisions.

## Business Value
- Flags SKUs that will run out **before the next delivery can arrive**
- Quantifies **revenue at risk** and **excess stock value**
- Recommends **how much to order** (EOQ-based, capped at 21 days of cover)
- Uses ABC analysis: high-revenue items get higher service levels (A 98%, B 95%, C 90%)

## Industry Relevance
Same logic family used by supermarkets and e-commerce players (demand planning, replenishment, safety stock). Relevant for Data Analyst, Business Analyst, Supply Chain Analyst and Data Science roles.

## Tech Stack
Python 3.10+, Pandas, NumPy, Scikit-learn (Linear Regression, Random Forest, Gradient Boosting), Matplotlib, Seaborn, Flask, Chart.js, Pytest.

## Architecture
```
Simulator -> Cleaning -> EDA charts -> Feature engineering -> Model training (3 models + baseline)
   -> 30-day recursive forecast -> Inventory optimisation -> CSV outputs -> Flask API + Dashboard
```

## Folder Structure
```
src/        config, data_generator, preprocessing, features, forecasting, inventory, visualization
app/        Flask API, templates, static (themed UI)
data/       raw/ and processed/ CSVs
outputs/    forecasts, recommendations, metrics
images/     charts and screenshots
models/     saved model
tests/      unit tests for inventory formulas
docs/       full project guide
runall.py   one-command pipeline + API
```

## Installation
```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```

## Dataset
Simulated: 3 stores x 12 products x 730 days (~26k rows). Includes weekly pattern, festive and summer seasonality, 12% growth trend, promotions, price changes, stock levels, stockout flags, plus injected duplicates and missing prices to demonstrate cleaning.

## How to Run
```bash
python runall.py              # pipeline + dashboard at http://127.0.0.1:5000
python runall.py --no-api     # pipeline only
python runall.py --skip-data  # reuse existing raw data
python -m pytest -q           # tests
```

## API
| Endpoint | Purpose |
|---|---|
| `/api/health` | service status |
| `/api/summary` | KPIs |
| `/api/recommendations?status=CRITICAL&store_id=S01` | replenishment table |
| `/api/forecast?store_id=S01&product_id=P001` | history + 30-day forecast |
| `/api/metrics` | model comparison |

## Simulation Workflow
1. Define catalog, stores, seasonality rules  2. Generate demand (Poisson + noise)  3. Simulate stock with an imperfect reorder policy (creates real stockouts)  4. Inject dirty data  5. Clean  6. Forecast  7. Optimise  8. Visualise.

## Results (example run)
| Model | WAPE (30-day holdout) |
|---|---|
| Seasonal Naive baseline | 32.9% |
| Linear Regression | 28.0% |
| Random Forest | 24.7% |
| **Gradient Boosting** | **24.4%** |

Evaluation is a true 30-day-ahead recursive forecast (not one-step-ahead). Per-store-per-product daily demand is noisy, so aggregate accuracy is much better than per-series accuracy.

## Screenshots
Charts are in `/images` (sales trend, seasonality heatmap, model comparison, actual vs predicted, 30-day forecast, inventory status, top reorders).

## Formulas
- Safety stock = z x sigma x sqrt(lead time)
- Reorder point = avg daily demand x lead time + safety stock
- EOQ = sqrt(2 x annual demand x order cost / holding cost)

## Limitations & Future Improvements
Synthetic data; no supplier constraints; promotions not forecasted. Next: multi-echelon stock, promo uplift model, price elasticity, anomaly detection, Prophet/XGBoost, Docker deployment.

## Learning Outcomes
Time-series feature engineering, leakage-safe lags, honest forecast evaluation, inventory theory, API design, dashboarding, Git workflow.

## Author
**Your Name** - [LinkedIn](https://linkedin.com/in/your-profile) - [GitHub](https://github.com/your-username)
