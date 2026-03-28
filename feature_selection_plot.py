import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('synthetic_fault_data_2017_2025.csv')

# Target (same as before)
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

features = ['Year', 'Month_Enc', 'Feeder_Enc', 'Weather_Factor', 'Fault_Location_km', 'Fault_Duration_s']
X = df[features]
y = df['Fault_Type_Num']

# RFE Feature Selection
model = RandomForestClassifier(n_estimators=200, random_state=42)
rfe = RFE(estimator=model, n_features_to_select=4)  # select top 4 like original report
rfe.fit(X, y)

# Create DataFrame for plotting
rfe_df = pd.DataFrame({
    'Feature': features,
    'Ranking': rfe.ranking_,
    'Selected': rfe.support_
})
rfe_df = rfe_df.sort_values('Ranking')

# Plot - exactly like the style in your original report
plt.figure(figsize=(10, 6))
colors = ['royalblue' if selected else 'lightgray' for selected in rfe_df['Selected']]
bars = plt.barh(rfe_df['Feature'], rfe_df['Ranking'], color=colors)

for i, bar in enumerate(bars):
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
             f"Rank {rfe_df['Ranking'].iloc[i]}", va='center', fontsize=10)

plt.title('Feature Selection using Recursive Feature Elimination (RFE)\n(Top 4 features selected)')
plt.xlabel('RFE Ranking (lower = more important)')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('Feature_Selection_RFE_Plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Feature_Selection_RFE_Plot.png saved")
print(rfe_df)