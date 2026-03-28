import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import joblib

# ====================== LOAD DATA ======================
df = pd.read_csv('synthetic_fault_data_2017_2025.csv')

# ====================== CREATE TARGET (as before) ======================
fault_cols = ['L_G_Fault', 'L_L_Fault', 'L_L_G_Fault', 'LLL_LLLG_Fault', 'Open_Circuits', 'Insulation_Failure']
df['Dominant_Fault'] = df[fault_cols].idxmax(axis=1)
df['Dominant_Count'] = df[fault_cols].max(axis=1)

fault_map = {'L_G_Fault': 2, 'L_L_Fault': 3, 'L_L_G_Fault': 4, 'LLL_LLLG_Fault': 5, 'Open_Circuits': 5, 'Insulation_Failure': 5}
df['Fault_Type_Num'] = df['Dominant_Fault'].map(fault_map)
df.loc[df['Dominant_Count'] == 0, 'Fault_Type_Num'] = 0

le_month = LabelEncoder()
le_feeder = LabelEncoder()
df['Month_Enc'] = le_month.fit_transform(df['Month'])
df['Feeder_Enc'] = le_feeder.fit_transform(df['Feeder'])

# ====================== FEATURES ======================
features = ['Year', 'Month_Enc', 'Feeder_Enc', 'Weather_Factor', 'Fault_Location_km', 'Fault_Duration_s']
X = df[features]
y = df['Fault_Type_Num']

# Time-based split
train_mask = df['Year'] <= 2023
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]

# ====================== TRAIN MODEL ======================
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# ====================== 1. FEATURE IMPORTANCE BAR CHART (exactly like your image) ======================
importances = pd.DataFrame({
    'Feature': features,
    'Importance (%)': model.feature_importances_ * 100
}).sort_values('Importance (%)', ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Importance (%)', y='Feature', data=importances, palette='Blues_d')
for i, v in enumerate(importances['Importance (%)']):
    ax.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=11, fontweight='bold')
plt.title('Feature Importance from Random Forest Classifier')
plt.xlabel('Feature Importance (%)')
plt.tight_layout()
plt.savefig('Feature-Importance_Barchart.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Feature-Importance_Barchart.png saved")

# ====================== 2. CORRELATION HEATMAP WITH VIF (exactly like your second image) ======================
X_const = add_constant(X_train)  # add constant for VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

corr_matrix = X_train.corr().round(4)

plt.figure(figsize=(12, 10))
ax = sns.heatmap(corr_matrix, annot=True, fmt=".4f", cmap="RdBu_r",
                 vmin=-1, vmax=1, linewidths=0.5, linecolor='white')

# Overlay VIF on diagonal (exactly like your image)
for i, feat in enumerate(corr_matrix.columns):
    vif_val = vif_data.loc[vif_data["Feature"] == feat, "VIF"].values[0]
    label = f"VIF\n{round(vif_val, 2)}" if vif_val < 1000 else "VIF\n∞"
    ax.text(i + 0.5, i + 0.5, label, ha='center', va='center',
            fontsize=11, fontweight='bold', color='white',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.85))

plt.title('Correlation Heatmap with VIF Overlaid on Diagonal\n(Real Data from synthetic_fault_data_2017_2025.csv)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('Correlation_Heatmap_with_VIF.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Correlation_Heatmap_with_VIF.png saved")

# ====================== SAVE MODEL & PREDICTIONS ======================
joblib.dump(model, 'random_forest_model.pkl')
df_test = df[~train_mask].copy()
df_test['Predicted_Fault_Type'] = y_pred
df_test.to_csv('real_predictions_2024_2025.csv', index=False)

print("\nAll done! Two plots saved:")
print("   • Feature-Importance_Barchart.png")
print("   • Correlation_Heatmap_with_VIF.png")