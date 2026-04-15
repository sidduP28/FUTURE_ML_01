# FUTURE_ML_01
# 📈 Store Sales Forecasting — Time Series ML

> Forecast daily sales across 54 stores and 33 product families for a large Ecuadorian grocery chain using advanced time series techniques.

---

## 📌 Project Overview

This project tackles the **Kaggle Store Sales - Time Series Forecasting** competition. The goal is to predict future sales using historical data enriched with calendar features, Fourier seasonality terms, and national/regional holiday indicators.

The pipeline uses `DeterministicProcess` for trend + seasonality modeling and `LinearRegression` as the base forecaster, with feature engineering as the primary performance driver.

---

## 🗂️ Dataset

| File | Description |
|------|-------------|
| `train.csv` | Historical sales by store, family, and date |
| `test.csv` | Dates to forecast |
| `stores.csv` | Store metadata (city, type, cluster) |
| `holidays_events.csv` | National & regional holidays |

> Source: [Kaggle — Store Sales Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

---

## ⚙️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14-purple)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green)

- **EDA** — Sales trends by day, month, year; rolling mean analysis
- **Trend Modeling** — `DeterministicProcess` (linear trend)
- **Seasonality** — `CalendarFourier` (Monthly order=4, Annual order=10)
- **Feature Engineering** — Weekend flag, week-of-year, holiday dummies
- **Forecasting** — `LinearRegression` with 200-day out-of-sample forecast
- **Analysis** — Periodogram for frequency component identification

---

## 🏗️ Project Structure

```
StoreSalesForecast/
│
├── ML_01_SALES_FORECAST.ipynb     # Main notebook
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── stores.csv
│   └── holidays_events.csv
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/StoreSalesForecast.git
cd StoreSalesForecast

# 2. Download dataset from Kaggle and place CSVs in /data
# https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch notebook
jupyter notebook ML_01_SALES_FORECAST.ipynb
```

---

## 📊 Key Features Built

| Feature | Description |
|---------|-------------|
| `trend` | Linear trend via DeterministicProcess |
| `sin/cos Fourier terms` | Weekly & annual seasonality cycles |
| `is_weekend` | Binary flag for Sat/Sun |
| `week_of_year` | Integer week number |
| `holiday_*` | One-hot encoded national/regional holidays |

---

## 📈 Visualizations Included

- Average sales by day of week, month, and year
- Rolling mean trend overlay on raw sales
- Periodogram — frequency component analysis
- Forecast plot: actuals vs predicted vs 200-day forecast
- Before/after trend removal comparison

---

## 🔍 Key Learnings

- Periodogram analysis is essential to identify dominant seasonal cycles before building Fourier features
- Holiday dummies for national events provide meaningful signal on residual errors
- `DeterministicProcess` cleanly separates trend from seasonality for linear models

---

## 📈 Suggested Improvements

- [ ] Add lag features (7-day, 14-day, 28-day) for autocorrelation signal
- [ ] Integrate oil price data as an external regressor
- [ ] Upgrade to a `BoostedHybrid` model (Linear trend + XGBoost residuals)
- [ ] Add payday flags (15th and last day of month for Ecuadorian payroll cycles)
- [ ] Evaluate with RMSLE metric to match competition scoring

---

## 👤 Author

**PITCHUKA ANJANI SIDDARTHA**
[LinkedIn](linkedin.com/in/pitchuka-anjani-siddartha-7a88593bb/?skipRedirect=true) · [Kaggle](https://www.kaggle.com/siddartha4)
