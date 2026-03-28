import pandas as pd
import joblib
from datetime import datetime, timedelta

print("=" * 70)
print("FUTURE PREDICTION")
print("Loading saved model and predicting Apr–Jun 2025 faults")
print("=" * 70)

# ─── Load saved model and scaler ────────────────────────────────────────────
model  = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
print("✅ Model loaded from 'model.joblib'")
print("✅ Scaler loaded from 'scaler.joblib'")

# ─── Load Test Data ─────────────────────────────────────────────────────────
test_df = pd.read_csv("test_dataset.csv")
print(f"Test data loaded: {len(test_df):,} samples (Apr–Jun 2025)")

# Add realistic timestamps
test_start = datetime(2025, 4, 1, 0, 0, 0)
test_df["Timestamp"] = [test_start + timedelta(minutes=15 * i)
                        for i in range(len(test_df))]

# ─── Apply same scaling, then predict ───────────────────────────────────────
features = ['Ia', 'Ib', 'Ic', 'Va']
X_test_scaled = scaler.transform(test_df[features])
test_df["Predicted_Fault_Type"] = model.predict(X_test_scaled)

# ─── Save for Streamlit Dashboard ───────────────────────────────────────────
test_df.to_csv("nigerian_test_data_with_predictions.csv", index=False)
print("✅ Predictions saved to 'nigerian_test_data_with_predictions.csv'")

# ─── Summary ────────────────────────────────────────────────────────────────
print("\n─── Predicted Fault-Type Distribution (Apr–Jun 2025) ───")
pred_dist = test_df['Predicted_Fault_Type'].value_counts().sort_index()
label_map = {
    0: 'Class 0 – Normal A', 1: 'Class 1 – Normal B',
    2: 'Class 2 – LG fault', 3: 'Class 3 – LL fault',
    4: 'Class 4 – LLG fault', 5: 'Class 5 – Healthy'
}

for cls, count in pred_dist.items():
    pct = 100 * count / len(test_df)
    print(f" {label_map.get(cls, cls):28s} {count:5,} ({pct:5.1f} %)")

print(f"\nTotal predictions generated: {len(test_df):,}")
print("\n🎉 Future predictions completed!")