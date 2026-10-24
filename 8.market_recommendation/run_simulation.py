import os
import pandas as pd
from recommendation_engine import MandiRecommender

model_path = r"..\7.model_training_and_evaluation\crop_price_prediction_model.joblib"
rec = MandiRecommender(model_path)
test_df = pd.read_csv(r"..\6.train_test_split\test_data.csv")
test_df['date'] = pd.to_datetime(test_df['date'])

sample_dates = test_df['date'].drop_duplicates().sample(20, random_state=42)
sim = []

for d in sample_dates:
    for crop in test_df['commodity'].unique():
        s = test_df[(test_df['date'] == d) & (test_df['commodity'] == crop)]
        if len(s) == 13:
            _, sm = rec.recommend(s, quantity_quintals=25)
            sim.append({
                "Date": str(d.date()),
                "Commodity": crop,
                "Best Mandi": sm['best_mandi'],
                "Worst Mandi": sm['worst_mandi'],
                "Net Realized (Best)": sm['net_best_price'],
                "Extra Profit (25 Quintals)": sm['profit_advantage']
            })

sim_df = pd.DataFrame(sim)
sim_df.to_csv("simulation_recommendation_results.csv", index=False)
print(f"Simulation executed: {len(sim_df)} scenarios.")
print(f"Avg extra profit on 25q: Rs. {sim_df['Extra Profit (25 Quintals)'].mean():,.2f}")
