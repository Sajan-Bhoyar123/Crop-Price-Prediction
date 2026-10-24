import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

out_dir = r"c:\Users\LOQ\Desktop\DSA CP\9.dashboard_and_visualizations"
os.makedirs(out_dir, exist_ok=True)
plots_dir = os.path.join(out_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'Arial'

# 1. Model Comparison Chart
print("Plot 1: Model Comparison...")
comp_df = pd.read_csv(r"c:\Users\LOQ\Desktop\DSA CP\7.model_training_and_evaluation\model_comparison_results.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#95a5a6', '#3498db', '#2ecc71', '#e67e22', '#9b59b6']
bars1 = ax1.bar(comp_df['Model'], comp_df['MAPE'], color=colors, width=0.55, edgecolor='black', alpha=0.9)
ax1.set_ylabel("Mean Absolute Percentage Error - MAPE (%)", fontsize=11, fontweight='bold')
ax1.set_title("Forecast Error Across Models (Lower is Better)", fontsize=13, fontweight='bold', pad=12)
ax1.set_xticklabels(comp_df['Model'], rotation=25, ha='right', fontsize=9.5)
ax1.set_ylim(0, max(comp_df['MAPE']) * 1.3)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

bars2 = ax2.bar(comp_df['Model'], comp_df['R2'], color=colors, width=0.55, edgecolor='black', alpha=0.9)
ax2.set_ylabel("R² Goodness-of-Fit Score", fontsize=11, fontweight='bold')
ax2.set_title("Variance Explained Across Models (Higher is Better)", fontsize=13, fontweight='bold', pad=12)
ax2.set_xticklabels(comp_df['Model'], rotation=25, ha='right', fontsize=9.5)
ax2.set_ylim(0.995, 1.0)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0003, f"{yval:.5f}", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

plt.suptitle("Benchmark Comparison: Machine Learning Models on Unseen Test Data (2026)", fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "model_accuracy_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

# 2. Per-Commodity Accuracy
print("Plot 2: Per-Commodity Accuracy...")
crop_df = pd.read_csv(r"c:\Users\LOQ\Desktop\DSA CP\7.model_training_and_evaluation\per_crop_evaluation.csv").sort_values("MAPE (%)")

plt.figure(figsize=(12, 6))
palette = sns.color_palette("mako", len(crop_df))
bars = plt.barh(crop_df['Commodity'], crop_df['MAPE (%)'], color=palette, edgecolor='black', height=0.65)
plt.xlabel("Mean Absolute Percentage Error - MAPE (%)", fontsize=11, fontweight='bold')
plt.ylabel("Commodity", fontsize=11, fontweight='bold')
plt.title("Price Forecasting Accuracy by Commodity (Test Period Jan–Sep 2026)\nAll Commodities Achieve MAPE < 2.5%", fontsize=13, fontweight='bold', pad=14)
plt.xlim(0, max(crop_df['MAPE (%)']) * 1.25)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.05, bar.get_y() + bar.get_height()/2.0, f"{w:.2f}% (R²={crop_df.loc[crop_df['MAPE (%)']==w, 'R2 Score'].values[0]:.4f})", ha='left', va='center', fontsize=9.5, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "per_crop_accuracy.png"), dpi=300, bbox_inches='tight')
plt.close()

# 3. Top 20 Feature Importance
print("Plot 3: Top 20 Feature Importance...")
feat_df = pd.read_csv(r"c:\Users\LOQ\Desktop\DSA CP\7.model_training_and_evaluation\feature_importance.csv").head(20).sort_values("Importance", ascending=True)

plt.figure(figsize=(11, 7.5))
palette = sns.color_palette("viridis", len(feat_df))
bars = plt.barh(feat_df['Feature'], feat_df['Importance'], color=palette, edgecolor='black', height=0.7)
plt.xlabel("LightGBM Feature Importance (Split Frequency)", fontsize=11, fontweight='bold')
plt.title("Top 20 Most Influential Predictors in Agricultural Price Formation\nDominant Drivers: Short-term Momentum, Supply Inflows, & Seasonality", fontsize=13, fontweight='bold', pad=14)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 5, bar.get_y() + bar.get_height()/2.0, f"{int(w)}", ha='left', va='center', fontsize=9, fontweight='bold')

plt.xlim(0, max(feat_df['Importance']) * 1.1)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "top20_feature_importance.png"), dpi=300, bbox_inches='tight')
plt.close()

# 4. Actual vs Predicted Time-Series Sample
print("Plot 4: Actual vs Predicted Time-Series...")
test_df = pd.read_csv(r"c:\Users\LOQ\Desktop\DSA CP\6.train_test_split\test_data.csv")
test_df['date'] = pd.to_datetime(test_df['date'])

# Load model and predict
bundle = joblib.load(r"c:\Users\LOQ\Desktop\DSA CP\7.model_training_and_evaluation\crop_price_prediction_model.joblib")
lgb_model = bundle['lgb_model']
xgb_model = bundle['xgb_model']
feature_cols = bundle['feature_cols']
clean_map = bundle['clean_map']

sub = test_df[(test_df['commodity'] == 'Cotton') & (test_df['market'] == 'Nagpur')].sort_values('date').copy()
sub_renamed = sub.rename(columns=clean_map)
lgb_d = lgb_model.predict(sub_renamed[feature_cols])
xgb_d = xgb_model.predict(sub_renamed[feature_cols])
sub['predicted'] = sub['modal_price'] + (0.5 * lgb_d + 0.5 * xgb_d)

plt.figure(figsize=(14, 5.5))
plt.plot(sub['date'], sub['target_next_day_modal_price'], label='Actual Next-Day Price (Ground Truth)', color='#2980b9', linewidth=2.2, alpha=0.9)
plt.plot(sub['date'], sub['predicted'], label='Ensemble Model Predicted Price', color='#e74c3c', linewidth=2, linestyle='--', alpha=0.9)
plt.fill_between(sub['date'], sub['target_next_day_modal_price'], sub['predicted'], color='#e74c3c', alpha=0.15, label='Residual Error Region')

plt.title("Ground Truth vs Predicted Modal Price: Cotton at Nagpur Mandi (Jan–Sep 2026)\nNear-Perfect Tracking with Mean Error < 1%", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Date", fontsize=11, fontweight='bold')
plt.ylabel("Modal Price (Rs. / Quintal)", fontsize=11, fontweight='bold')
plt.legend(frameon=True, facecolor='white', loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "actual_vs_predicted_timeseries.png"), dpi=300, bbox_inches='tight')
plt.close()

# 5. Market Profit Advantage Simulation Distribution
print("Plot 5: Market Profit Advantage...")
sim_df = pd.read_csv(r"c:\Users\LOQ\Desktop\DSA CP\8.market_recommendation\simulation_recommendation_results.csv")

plt.figure(figsize=(10, 5))
sns.histplot(sim_df['Extra Profit (25 Quintals)'], bins=25, kde=True, color='#27ae60', edgecolor='black', alpha=0.7)
mean_val = sim_df['Extra Profit (25 Quintals)'].mean()
plt.axvline(mean_val, color='#c0392b', linestyle='--', linewidth=2.5, label=f"Mean Gain: Rs. {mean_val:,.2f}")
plt.title("Distribution of Extra Profit Earned per Farmer (25 Quintals Harvest)\nUsing Market Recommendation Engine vs Sub-Optimal Mandis", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Additional Revenue Earned (Rs.)", fontsize=11, fontweight='bold')
plt.ylabel("Scenario Frequency", fontsize=11, fontweight='bold')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "market_profit_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()

# 6. Raw Predicted Price Across Mandis (1 Quintal - Before Transport)
print("Plot 6: Raw Predicted Price Across Mandis (1 Quintal)...")
sample_slice = test_df[(test_df['date'] == '2026-05-15') & (test_df['commodity'] == 'Onion')].copy()
sample_renamed = sample_slice.rename(columns=clean_map)
lgb_delta = lgb_model.predict(sample_renamed[feature_cols])
xgb_delta = xgb_model.predict(sample_renamed[feature_cols])
sample_slice['predicted_modal_price'] = sample_slice['modal_price'] + (0.5 * lgb_delta + 0.5 * xgb_delta)
raw_rank = sample_slice.sort_values('predicted_modal_price', ascending=False).reset_index(drop=True)

plt.figure(figsize=(11, 5))
colors_raw = ['#2e7d32' if i == 0 else '#c62828' if i == len(raw_rank)-1 else '#1976d2' for i in range(len(raw_rank))]
bars_r = plt.bar(raw_rank['market'], raw_rank['predicted_modal_price'], color=colors_raw, edgecolor='black', alpha=0.85)
plt.ylabel("Predicted Modal Price (Rs. / quintal)", fontsize=11, fontweight='bold')
plt.title("Predicted Modal Price Across Mandis: Onion on 15-May-2026 (1 Quintal — Without Transport Deduction)", fontsize=12, fontweight='bold', pad=12)
plt.xticks(rotation=30, ha='right')
plt.ylim(min(raw_rank['predicted_modal_price']) * 0.95, max(raw_rank['predicted_modal_price']) * 1.05)
for b in bars_r:
    h = b.get_height()
    plt.text(b.get_x() + b.get_width()/2.0, h + 15, f"Rs. {h:,.0f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "mandi_raw_predicted_price_1quintal.png"), dpi=300, bbox_inches='tight')
plt.close()

print("All publication-quality visualization plots successfully generated!")
