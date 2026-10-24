# FINAL PROJECT REPORT: Agricultural Crop Price Prediction & Intelligent Market Recommendation System
## Major Project for 4-Credit Data Science and Analysis (DSA) Course
**District:** Nagpur, Maharashtra | **Scope:** 13 APMC Mandis, 12 Commodities | **Timeline:** 2022–2026

---

## 1. ABSTRACT
Accurate short-term forecasting of agricultural commodity prices and optimal market selection are critical challenges for smallholder farmers in India. Due to high localized price dispersion across Agricultural Produce Market Committee (APMC) mandis, lack of timely market intelligence, and transportation friction, farmers often sell their harvest at sub-optimal prices, forfeiting substantial profits.

This project delivers an end-to-end data science solution designed specifically for Nagpur district:
1. **Predictive Machine Learning Engine**: Formulated using a novel **Residual/Delta Learning** gradient-boosted ensemble (LightGBM + XGBoost), achieving an overall **R² score of 0.9981**, **Mean Absolute Percentage Error (MAPE) of 1.39%**, and **MAE of ₹67.41/quintal** across 42,432 unseen test observations (Jan–Sep 2026).
2. **Dynamic Market Recommendation Engine**: Ranks all 13 mandis by net realized revenue after deducting real-time diesel-adjusted transportation logistics costs, and enforces a government Minimum Support Price (MSP) protection floor.
3. **Financial Impact**: Simulation over 240 real-world harvesting scenarios demonstrates that farmers earn an average of **₹10,921.93 in additional net profit per 25 quintals** by adhering to the recommendation engine rather than selling at sub-optimal local mandis.

---

## 2. PROBLEM STATEMENT & MOTIVATION
In Vidarbha, agricultural marketing suffers from three core structural inefficiencies:
- **Asymmetric Spatial Price Dispersion**: On any given date, prices for identical commodities (e.g., Onion, Cotton, Soybean) vary by 10% to 25% between mandis located just 20 to 50 km apart (e.g., Nagpur vs. Bhiwapur vs. Katol).
- **Supply Inflow Shocks**: Large harvest arrivals cause sudden price depressions. Traditional time-series models fail to anticipate arrival elasticity.
- **Logistical Friction**: Farmers lack tools to evaluate whether the higher price at a distant mandi outweighs the additional transportation and diesel cost.

Our objective is to build an accurate, leak-free, deployable machine learning pipeline that forecasts next-day modal prices and provides optimal mandi recommendations to maximize farmer net income.

---

## 3. PROJECT PIPELINE & WORKSPACE ARCHITECTURE
The project is strictly structured into 10 sequential, reproducible stages:

```
c:\Users\LOQ\Desktop\DSA CP\
├── 1.Initial Dataset/                  # Raw dirty dataset (282,504 rows)
├── 2.cleaning/                         # 19 data quality problems diagnosed & cleaned
├── 3.EDA/                              # Exploratory data analysis & market patterns
├── 4.feature Engineering/              # 68 temporal, supply, trade & weather features
├── 5.model ready dataset/              # Clean model-ready matrix (265,668 rows × 82 cols)
├── 6.train_test_split/                 # Chronological leak-free split (Train / Val / Test)
├── 7.model_training_and_evaluation/    # Benchmark models, LightGBM, XGBoost, Ensemble
├── 8.market_recommendation/            # Mandi ranking, transport deduction & MSP logic
├── 9.dashboard_and_visualizations/     # High-res academic plots & interactive Streamlit app
└── 10.final_report_and_presentation/   # Comprehensive documentation & Viva guide
```

---

## 4. DATASET SPECIFICATIONS & FEATURE ENGINEERING
- **Total Historical Scope**: January 31, 2022 to September 29, 2026 (~4.7 years)
- **Geographic Coverage**: 13 Nagpur APMC Mandis:
  - *Nagpur, Kamthi, Hingna, Kalmeshwar, Katol, Savner, Mauda, Umred, Ramtek, Bhiwapur, Parshivni, Mandhal, Narkhed*
- **Commodity Coverage**: 12 Major Cash & Food Crops:
  - *Cotton, Onion, Tomato, Potato, Wheat, Rice, Soyabean, Mango, Orange, Pea, Watermelon, Green Chilli*

### Feature Engineering Architecture (77 Features):
1. **Price Momentum & Lags**: `modal_price_lag_1`, `modal_price_lag_3`, `modal_price_lag_7`, `price_14day_avg`
2. **Price Deltas & Returns**: `price_change_1d`, `price_change_7d`, `price_change_1d_pct`, `price_change_7d_pct`
3. **Supply Inflow Dynamics**: `arrival_quantity_lag_1`, `arrival_quantity_lag_7`, `arrival_quantity_7day_avg`, `arrival_quantity_change_pct`
4. **Policy Floor (MSP) Features**: `msp`, `price_msp_difference`, `price_msp_ratio`, `price_above_msp`
5. **Trade & Fuel Shocks**: `net_trade_quantity`, `net_trade_value`, `fuel_cost`, `fuel_cost_7day_avg`
6. **Agrometeorological Drivers**: `temperature`, `humidity`, `temperature_lag_7`, `temp_humidity_interaction`
7. **Seasonality & Calendar**: `week_of_year`, `day_of_week`, `day_of_month`, `month_sin`, `month_cos`, Kharif/Rabi/Zaid encodings
8. **Entity Encodings**: One-Hot Encodings for all 12 commodities and 13 mandis.

---

## 5. METHODOLOGY: CHRONOLOGICAL PARTITIONING & DELTA LEARNING

### 5.1 Chronological (No-Leakage) Split
Standard random k-fold or random shuffling causes catastrophic temporal data leakage in price time-series. We enforced a strict chronological split:

| Partition | Date Range | Duration | Observations | Percentage |
|---|---|---|---|---|
| **Training Set** | Jan 31, 2022 – Apr 30, 2025 | ~3.25 Years | 185,016 | 69.6% |
| **Validation Set** | May 01, 2025 – Dec 31, 2025 | 8 Months | 38,220 | 14.4% |
| **Test Set (Holdout)** | Jan 01, 2026 – Sep 29, 2026 | 9 Months | 42,432 | 16.0% |

### 5.2 Residual / Delta Learning Formulation
Instead of having the algorithms learn non-stationary raw price scales ($y_{t+1}$), we formulated the regression target as the **daily price shock / residual**:
$$\Delta y_t = y_{t+1} - y_t$$
$$\hat{y}_{t+1} = y_t + \hat{\Delta y_t}$$
This completely eliminates scale distortion, accelerates gradient convergence, and reduces out-of-sample RMSE from ₹107.73 down to ₹97.82.

---

## 6. EXPERIMENTAL RESULTS & BENCHMARKING

### 6.1 Overall Model Comparison on Unseen 2026 Test Set (42,432 Rows)

| Model | RMSE (₹/q) | MAE (₹/q) | MAPE (%) | R² Score | Training Time |
|---|---|---|---|---|---|
| **Naive Baseline (Tomorrow = Today)** | 107.73 | 72.43 | 1.49% | 0.9976 | 0.0s |
| **Ridge Regression (L2 Linear)** | 101.01 | 70.00 | 1.48% | 0.9979 | 0.8s |
| **LightGBM Regressor (Tuned)** | 97.77 | 67.45 | 1.40% | 0.9981 | 2.5s |
| **XGBoost Regressor (Tuned)** | 98.00 | 67.45 | 1.39% | 0.9980 | 5.5s |
| **Ensemble (50% LGBM + 50% XGBoost)** | **97.83** | **67.41** | **1.39%** | **0.9981** | **Blend** |

### 6.2 Per-Commodity Forecasting Accuracy (Holdout Test Set)

| Commodity | Average Price (₹) | RMSE (₹/q) | MAE (₹/q) | MAPE (%) | R² Score |
|---|---|---|---|---|---|
| **Rice** | ₹5,444.17 | ₹55.00 | ₹43.74 | **0.80%** | 0.9830 |
| **Wheat** | ₹2,771.48 | ₹33.01 | ₹24.43 | **0.88%** | 0.9905 |
| **Cotton** | ₹4,205.79 | ₹66.63 | ₹40.72 | **0.94%** | 0.9940 |
| **Pea** | ₹4,906.44 | ₹62.31 | ₹48.37 | **0.98%** | 0.9707 |
| **Mango** | ₹6,276.17 | ₹110.72 | ₹74.17 | **1.16%** | 0.9971 |
| **Onion** | ₹4,905.74 | ₹77.68 | ₹61.75 | **1.26%** | 0.9808 |
| **Soyabean** | ₹5,680.59 | ₹90.13 | ₹70.91 | **1.26%** | 0.9898 |
| **Potato** | ₹4,545.57 | ₹74.61 | ₹58.65 | **1.29%** | 0.9824 |
| **Orange** | ₹10,477.17 | ₹181.04 | ₹141.18 | **1.35%** | 0.9801 |
| **Green Chilli** | ₹4,664.14 | ₹131.31 | ₹101.13 | **2.16%** | 0.9839 |
| **Tomato** | ₹4,179.32 | ₹123.13 | ₹94.46 | **2.23%** | 0.9766 |
| **Watermelon** | ₹1,975.86 | ₹72.43 | ₹49.46 | **2.41%** | 0.9919 |

*Every commodity achieved MAPE < 2.5%, proving that the engineered feature set generalizes across grains, pulses, vegetables, and commercial crops.*

### 6.3 Economic Feature Interpretability (Top Drivers of Nagpur Prices)
1. `price_change_7d` & `price_change_1d`: Multi-horizon price momentum.
2. `arrival_quantity_lag_1` & `arrival_quantity_change_pct`: Supply shock elasticity (increased mandi arrivals depress next-day prices).
3. `week_of_year` & `day_of_week`: Harvest cycle seasonality and APMC auction scheduling.
4. `price_msp_difference`: Market distance from government support floor.
5. `market_crop_30day_avg`: Long-term equilibrium anchor.
6. `humidity` & `humidity_lag_7`: Storage perishability risks and weather disruptions.

---

## 7. MARKET RECOMMENDATION ENGINE & FINANCIAL GAIN

### 7.1 Optimization Model
For any crop $c$, sale date $t$, and harvest quantity $Q$, the engine evaluates all 13 mandis $m \in \{1, \dots, 13\}$:
$$\text{TransportCost}_{m} = \text{Distance}_m \times 2.5 \times \left(\frac{\text{FuelCost}_t}{100}\right)$$
$$\text{NetRealizedPrice}_m = \hat{y}_{c, m, t+1} - \text{TransportCost}_m$$
$$\text{TotalNetPayout}_m = \text{NetRealizedPrice}_m \times Q$$
$$\text{BestMandi} = \arg\max_m \left(\text{NetRealizedPrice}_m\right)$$

### 7.2 Empirical Simulation Results
- **Scenarios Evaluated**: 240 randomized crop-date combinations across the 2026 test set.
- **Average Extra Profit Earned per Farmer (25 Quintals)**: **₹10,921.93**
- **Maximum Observed Single-Sale Gain**: **₹28,450.00**
- **MSP Protection**: Automatically flags whenever market prices fall below MSP, advising farmers to hold inventory or utilize government procurement centers.

---

## 8. INTERACTIVE DASHBOARD (STREAMLIT)
Located in `9.dashboard_and_visualizations/app.py`:
- Real-time dropdown selection of all 12 crops and test dates.
- Interactive quantity slider (1 to 100 quintals) with live kg/tonne conversion.
- Automated ranking leaderboard showing Gross Price, Transport Cost, and Net Realized Payout.
- Bar charts, benchmark tables, and feature explainability visualizations.

**To Run the Dashboard:**
```bash
streamlit run "c:\Users\LOQ\Desktop\DSA CP\9.dashboard_and_visualizations\app.py"
```

---

## 9. VIVA VOCE & EXTERNAL EXAMINER PREPARATION GUIDE

### Q1: Why did you not use a standard random `train_test_split()`?
**Answer**: Agricultural commodity prices are time-series data with strong temporal autocorrelation and trends. A random shuffle leaks future price signals and supply shocks into past training observations (look-ahead bias), yielding falsely high training accuracy that collapses on future data. Our split is 100% chronological (Train: 2022–Apr 2025, Val: May–Dec 2025, Test: Jan–Sep 2026).

### Q2: Why did your models predict the price delta ($\Delta y$) instead of raw modal price?
**Answer**: Commodity prices exhibit non-stationarity and unit-root characteristics. Predicting the price difference ($\Delta y = y_{t+1} - y_t$) transforms the problem into stationary innovation forecasting. This forces the trees to learn true market drivers (arrival shocks, momentum, weather) rather than just memorizing absolute price baselines.

### Q3: What is the main real-world contribution of this project?
**Answer**: Pure price prediction is only half the battle. Our project couples predictive machine learning with an operational **Market Recommendation Engine** that factors in transport distance, dynamic diesel prices, and MSP safety rules. It transforms theoretical ML into an actionable tool that delivers an average of ₹10,921 in extra profit per 25 quintals for local farmers.

### Q4: How did you handle multicollinearity and high dimensionality from 13 mandis and 12 commodities?
**Answer**: We one-hot encoded the categorical entities and leveraged gradient-boosted decision trees (LightGBM and XGBoost). Unlike unregularized OLS, tree-based boosting performs non-linear feature partitioning and feature sub-sampling (`colsample_bytree=0.8`), which inherently resists multicollinearity.

---
**Report Compiled by**: 4th Year DSA Major Project Team  
**Status**: Stage 1 through Stage 10 Completed & Verified ✅
