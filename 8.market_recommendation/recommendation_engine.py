"""
Agricultural Market Recommendation Engine
Nagpur District Mandis (2022-2026)
"""

import pandas as pd
import numpy as np
import joblib

class MandiRecommender:
    MANDI_DISTANCES = {
        'Nagpur': 5,
        'Kamthi': 18,
        'Hingna': 15,
        'Kalmeshwar': 22,
        'Katol': 58,
        'Savner': 38,
        'Mauda': 35,
        'Umred': 45,
        'Ramtek': 48,
        'Bhiwapur': 68,
        'Parshivni': 42,
        'Mandhal': 62,
        'Narkhed': 85
    }

    def __init__(self, model_bundle_path):
        bundle = joblib.load(model_bundle_path)
        self.lgb_model = bundle["lgb_model"]
        self.xgb_model = bundle["xgb_model"]
        self.feature_cols = bundle["feature_cols"]
        self.clean_map = bundle["clean_map"]

    def recommend(self, df_mandi_rows, quantity_quintals=10, custom_origin=None):
        """
        Given rows of all mandis for a specific date and commodity,
        predicts tomorrow's price, calculates transport cost and net profit,
        and returns a sorted ranking.
        """
        df = df_mandi_rows.copy().rename(columns=self.clean_map)

        # Delta prediction
        lgb_delta = self.lgb_model.predict(df[self.feature_cols])
        xgb_delta = self.xgb_model.predict(df[self.feature_cols])
        ens_delta = 0.5 * lgb_delta + 0.5 * xgb_delta

        df['predicted_modal_price'] = df['modal_price'] + ens_delta

        # Fuel cost and distance
        fuel_rate = df['fuel_cost'].iloc[0] if 'fuel_cost' in df.columns else 95.0
        df['distance_km'] = df['market'].map(self.MANDI_DISTANCES).fillna(35.0)

        # Transport cost formula: (Distance * 2.5 * (fuel_rate / 100))
        df['transport_cost_per_q'] = (df['distance_km'] * 2.5 * (fuel_rate / 100.0)).round(2)
        df['net_realized_price'] = (df['predicted_modal_price'] - df['transport_cost_per_q']).round(2)
        df['total_net_payout'] = (df['net_realized_price'] * quantity_quintals).round(2)

        avg_price = df['predicted_modal_price'].mean()
        df['gain_pct_vs_avg'] = (((df['predicted_modal_price'] - avg_price) / avg_price) * 100).round(2)

        # Actual ground truth price and forecasting accuracy
        if 'target_next_day_modal_price' in df.columns:
            df['actual_price'] = df['target_next_day_modal_price'].round(2)
            df['accuracy_pct'] = ((1.0 - (abs(df['actual_price'] - df['predicted_modal_price']) / df['actual_price'])) * 100).round(2)

        # Check MSP floor safety
        msp = df['msp'].iloc[0] if 'msp' in df.columns else 0
        df['msp_value'] = msp
        df['is_above_msp'] = df['predicted_modal_price'] >= msp

        # Ranking
        ranking = df.sort_values('net_realized_price', ascending=False).reset_index(drop=True)
        ranking.index += 1

        best = ranking.iloc[0]
        worst = ranking.iloc[-1]
        profit_advantage = best['total_net_payout'] - worst['total_net_payout']

        summary = {
            "best_mandi": best['market'],
            "predicted_best_price": best['predicted_modal_price'],
            "net_best_price": best['net_realized_price'],
            "total_best_payout": best['total_net_payout'],
            "worst_mandi": worst['market'],
            "total_worst_payout": worst['total_net_payout'],
            "profit_advantage": profit_advantage,
            "msp": msp,
            "is_above_msp": bool(best['is_above_msp']),
            "quantity_quintals": quantity_quintals
        }

        return ranking, summary
