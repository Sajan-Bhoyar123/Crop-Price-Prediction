import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="KrishiPrice AI — Crop Price & Market Recommender",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1b5e20;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 20px;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2e7d32;
        margin-bottom: 10px;
    }
    .highlight-profit {
        color: #2e7d32;
        font-size: 1.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Cache data and model loading
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "..", "7.model_training_and_evaluation", "crop_price_prediction_model.joblib")
    bundle = joblib.load(model_path)
    
    test_path = os.path.join(base_dir, "..", "6.train_test_split", "test_data.csv")
    test_df = pd.read_csv(test_path)
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    return bundle, test_df

bundle, test_df = load_assets()
lgb_model = bundle['lgb_model']
xgb_model = bundle['xgb_model']
feature_cols = bundle['feature_cols']
clean_map = bundle['clean_map']

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

# Sidebar Controls
st.sidebar.title("🌾 Harvest Advisory Controls")
st.sidebar.markdown("Configure crop and harvest volume to identify the most profitable APMC mandi in Nagpur district.")

commodities = sorted(test_df['commodity'].unique())
selected_crop = st.sidebar.selectbox("Select Commodity", commodities, index=commodities.index("Onion") if "Onion" in commodities else 0)

available_dates = test_df[test_df['commodity'] == selected_crop]['date'].dt.date.unique()
available_dates = sorted(available_dates, reverse=True)
selected_date = st.sidebar.selectbox("Select Sale Date", available_dates, index=min(10, len(available_dates)-1))

quantity = st.sidebar.number_input("Harvest Quantity (Quintals)", min_value=1.0, max_value=1000.0, value=25.0, step=5.0)
st.sidebar.markdown(f"*{quantity*100:.0f} kg / {quantity/10:.1f} Metric Tonnes*")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Transport Model")
st.sidebar.info("Transport deductions use dynamic fuel rates multiplied by round-trip mandi distances.")

# Header
st.markdown('<div class="main-header">🌾 KrishiPrice AI: Nagpur Mandi Price Predictor & Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">4-Credit Data Science Major Project | Machine Learning Driven Agricultural Market Intelligence</div>', unsafe_allow_html=True)

# Filter test slice
slice_df = test_df[(test_df['date'].dt.date == selected_date) & (test_df['commodity'] == selected_crop)].copy()

if len(slice_df) == 0:
    st.warning("No mandi reporting data available for the chosen date and crop combination.")
else:
    # Run predictions
    slice_renamed = slice_df.rename(columns=clean_map)
    lgb_pred = lgb_model.predict(slice_renamed[feature_cols])
    xgb_pred = xgb_model.predict(slice_renamed[feature_cols])
    delta_pred = 0.5 * lgb_pred + 0.5 * xgb_pred

    slice_df['predicted_modal_price'] = slice_df['modal_price'] + delta_pred
    fuel_rate = slice_df['fuel_cost'].iloc[0] if 'fuel_cost' in slice_df.columns else 95.0
    slice_df['distance_km'] = slice_df['market'].map(MANDI_DISTANCES).fillna(35.0)
    slice_df['transport_cost_per_q'] = (slice_df['distance_km'] * 2.5 * (fuel_rate / 100.0)).round(2)
    slice_df['net_realized_price'] = (slice_df['predicted_modal_price'] - slice_df['transport_cost_per_q']).round(2)
    slice_df['total_net_payout'] = (slice_df['net_realized_price'] * quantity).round(2)
    
    avg_price = slice_df['predicted_modal_price'].mean()
    slice_df['gain_pct_vs_avg'] = (((slice_df['predicted_modal_price'] - avg_price) / avg_price) * 100).round(2)
    
    msp = slice_df['msp'].iloc[0] if 'msp' in slice_df.columns else 0
    slice_df['is_above_msp'] = slice_df['predicted_modal_price'] >= msp

    ranking = slice_df.sort_values('net_realized_price', ascending=False).reset_index(drop=True)
    best = ranking.iloc[0]
    worst = ranking.iloc[-1]
    extra_profit = best['total_net_payout'] - worst['total_net_payout']

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="🏆 Top Recommended Mandi", value=f"{best['market']} Mandi")
    with col2:
        st.metric(label="💵 Net Price / Quintal", value=f"₹{best['net_realized_price']:,.2f}", delta=f"{best['gain_pct_vs_avg']:+.1f}% vs Avg")
    with col3:
        st.metric(label="💰 Expected Net Payout", value=f"₹{best['total_net_payout']:,.2f}")
    with col4:
        st.metric(label="📈 Extra Profit vs Lowest Mandi", value=f"₹{extra_profit:,.2f}", delta=f"+₹{extra_profit/quantity:,.1f}/quintal")

    # MSP Safety Banner
    if msp > 0:
        if best['is_above_msp']:
            st.success(f"✅ **MSP Protected**: Predicted market price (₹{best['predicted_modal_price']:,.2f}) is ABOVE government MSP (₹{msp:,.2f}/quintal). Safe to sell.")
        else:
            st.error(f"⚠️ **Below MSP Warning**: Predicted market price (₹{best['predicted_modal_price']:,.2f}) is BELOW government MSP (₹{msp:,.2f}/quintal). Advisory: Farmer should store produce in warehouse or sell through NAFED/Govt Procurement Centres.")

    # Tabs for visualization
    tab1, tab2, tab3 = st.tabs(["📊 Market Comparison & Ranking", "📈 Model Accuracy & Benchmarks", "🔍 Explainability & Features"])

    with tab1:
        st.markdown(f"### Mandi Price Comparison for {selected_crop} on {selected_date}")

        # View Selector for flexibility
        graph_view = st.radio(
            "Select Graph View:",
            ["Both Graphs (Side-by-Side Comparison)", "1️⃣ Raw Predicted Price (1 Quintal - Without Transport Cost)", "2️⃣ Net Realized Price (After Transport Deductions)"],
            horizontal=True
        )

        # 1. Raw Predicted Price Ranking (1 Quintal, Before Transport Deductions)
        raw_ranking = slice_df.sort_values('predicted_modal_price', ascending=False).reset_index(drop=True)
        raw_best = raw_ranking.iloc[0]
        raw_worst = raw_ranking.iloc[-1]

        def plot_raw_chart():
            fig_raw, ax_raw = plt.subplots(figsize=(10, 4.5))
            colors_raw = ['#2e7d32' if m == raw_best['market'] else '#c62828' if m == raw_worst['market'] else '#1976d2' for m in raw_ranking['market']]
            bars_raw = ax_raw.bar(raw_ranking['market'], raw_ranking['predicted_modal_price'], color=colors_raw, edgecolor='black', alpha=0.85)
            ax_raw.set_ylabel("Predicted Modal Price (₹ / quintal)", fontsize=10, fontweight='bold')
            ax_raw.set_title(f"Predicted Price by Mandi for {selected_crop} (1 Quintal — Without Transport Deductions)", fontsize=11, fontweight='bold')
            plt.setp(ax_raw.get_xticklabels(), rotation=30, ha='right')
            ax_raw.set_ylim(min(raw_ranking['predicted_modal_price']) * 0.95, max(raw_ranking['predicted_modal_price']) * 1.05)
            for b in bars_raw:
                h = b.get_height()
                ax_raw.text(b.get_x() + b.get_width()/2.0, h + 15, f"₹{h:,.0f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_raw)

        def plot_net_chart():
            fig_net, ax_net = plt.subplots(figsize=(10, 4.5))
            colors_net = ['#2e7d32' if m == best['market'] else '#c62828' if m == worst['market'] else '#1976d2' for m in ranking['market']]
            bars_net = ax_net.bar(ranking['market'], ranking['net_realized_price'], color=colors_net, edgecolor='black', alpha=0.85)
            ax_net.set_ylabel("Net Realized Price (₹ / quintal)", fontsize=10, fontweight='bold')
            ax_net.set_title(f"Net Realized Price by Mandi (After Transport Deductions)", fontsize=11, fontweight='bold')
            plt.setp(ax_net.get_xticklabels(), rotation=30, ha='right')
            ax_net.set_ylim(min(ranking['net_realized_price']) * 0.95, max(ranking['net_realized_price']) * 1.05)
            for b in bars_net:
                h = b.get_height()
                ax_net.text(b.get_x() + b.get_width()/2.0, h + 15, f"₹{h:,.0f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_net)

        if graph_view == "1️⃣ Raw Predicted Price (1 Quintal - Without Transport Cost)":
            st.markdown("#### 🌾 Predicted Modal Price (1 Quintal — Without Transport Cost)")
            plot_raw_chart()
        elif graph_view == "2️⃣ Net Realized Price (After Transport Deductions)":
            st.markdown("#### 🚚 Net Realized Price (After Transport & Fuel Deductions)")
            plot_net_chart()
        else:
            st.markdown("#### 1️⃣ Predicted Modal Price (1 Quintal — Without Transport Cost)")
            plot_raw_chart()
            st.markdown("#### 2️⃣ Net Realized Price (After Distance & Fuel Cost Deductions)")
            plot_net_chart()

        # Insight banner showing the difference
        if raw_best['market'] != best['market']:
            st.info(f"💡 **Logistics Impact:** Notice that **{raw_best['market']}** offers the highest raw price (₹{raw_best['predicted_modal_price']:,.2f}), but after transport deductions, **{best['market']}** yields the highest net in-hand profit (₹{best['net_realized_price']:,.2f}/quintal)!")
        else:
            st.info(f"💡 **Consistent Leader:** **{best['market']} Mandi** offers both the highest raw price (₹{raw_best['predicted_modal_price']:,.2f}) and highest net profit after transport!")

        # Compute actual price and accuracy percentage for each mandi
        ranking['actual_price'] = ranking['target_next_day_modal_price']
        ranking['accuracy_pct'] = (1.0 - (abs(ranking['actual_price'] - ranking['predicted_modal_price']) / ranking['actual_price'])) * 100
        avg_date_accuracy = ranking['accuracy_pct'].mean()
        avg_date_error = abs(ranking['actual_price'] - ranking['predicted_modal_price']).mean()

        # Table Display
        st.markdown("#### Detailed Mandi Realization Breakdown")
        st.caption(f"🎯 **Model Evaluation for Selected Date:** Mean Forecasting Accuracy = **{avg_date_accuracy:.2f}%** | Mean Absolute Error = **±₹{avg_date_error:.2f}/quintal**")
        
        table_df = ranking[['market', 'actual_price', 'predicted_modal_price', 'accuracy_pct', 'distance_km', 'transport_cost_per_q', 'net_realized_price', 'total_net_payout', 'gain_pct_vs_avg']].copy()
        table_df.columns = [
            'Mandi', 
            'Actual Price (₹/q)', 
            'Predicted Price Without Transport (₹/q)', 
            'Accuracy (%)', 
            'Distance (km)', 
            'Transport Cost (₹/q)', 
            'Net Realized Price (₹/q)', 
            f'Total Net Payout on {quantity:.0f}q (₹)', 
            'Advantage vs Avg (%)'
        ]
        st.dataframe(table_df.style.format({
            'Actual Price (₹/q)': '₹{:,.2f}',
            'Predicted Price Without Transport (₹/q)': '₹{:,.2f}',
            'Accuracy (%)': '{:.2f}%',
            'Transport Cost (₹/q)': '₹{:,.2f}',
            'Net Realized Price (₹/q)': '₹{:,.2f}',
            f'Total Net Payout on {quantity:.0f}q (₹)': '₹{:,.2f}',
            'Advantage vs Avg (%)': '{:+.2f}%'
        }), use_container_width=True)

    with tab2:
        st.markdown("### Machine Learning Model Benchmarks on Unseen Future Data (2026)")
        col_a, col_b = st.columns(2)
        with col_a:
            comp_path = os.path.join(os.path.dirname(__file__), "..", "7.model_training_and_evaluation", "model_comparison_results.csv")
            comp_df = pd.read_csv(comp_path)
            st.dataframe(comp_df.style.highlight_min(subset=['RMSE', 'MAE', 'MAPE'], color='#c8e6c9').highlight_max(subset=['R2'], color='#c8e6c9'), use_container_width=True)
            st.markdown("""
            **Key Engineering Notes:**
            - Models evaluated chronologically on 42,432 holdout records.
            - **Residual / Delta Learning** achieved R² = **0.9981** and MAPE = **1.39%**.
            """)
        with col_b:
            crop_path = os.path.join(os.path.dirname(__file__), "..", "7.model_training_and_evaluation", "per_crop_evaluation.csv")
            crop_df = pd.read_csv(crop_path)
            st.dataframe(crop_df.style.format({
                'Avg Price (Rs)': '₹{:,.2f}',
                'RMSE (Rs)': '₹{:,.2f}',
                'MAE (Rs)': '₹{:,.2f}',
                'MAPE (%)': '{:.2f}%',
                'R2 Score': '{:.4f}'
            }), use_container_width=True)

    with tab3:
        st.markdown("### Feature Importance & Economic Drivers")
        feat_path = os.path.join(os.path.dirname(__file__), "..", "7.model_training_and_evaluation", "feature_importance.csv")
        feat_df = pd.read_csv(feat_path).head(15)
        
        fig_feat, ax_f = plt.subplots(figsize=(9, 4.5))
        ax_f.barh(feat_df['Feature'][::-1], feat_df['Importance'][::-1], color='#1b5e20', edgecolor='black')
        ax_f.set_xlabel("Feature Split Frequency", fontsize=10, fontweight='bold')
        ax_f.set_title("Top 15 Most Influential Features Driving Price Changes", fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_feat)
