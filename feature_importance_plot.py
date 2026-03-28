import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# ─── Load data ───
df = pd.read_csv('train_dataset.csv')

all_features      = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
selected_features = ['Ia', 'Ib', 'Ic', 'Va']

X = df[all_features]
y = df['Fault_Type']

# ─── Scale and fit RF ───
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_scaled, y)

# ─── Build importance DataFrame ───
importance_df = pd.DataFrame({
    'Feature'   : all_features,
    'Importance': (rf.feature_importances_ * 100).round(1)
}).sort_values('Importance', ascending=False).reset_index(drop=True)

importance_df['Rank']     = range(1, len(importance_df) + 1)
importance_df['Selected'] = importance_df['Feature'].apply(
    lambda f: 'Yes' if f in selected_features else 'No'
)

# ─── Print table to terminal ───
print("\n=== Feature Importance Summary Table ===")
print(importance_df[['Rank', 'Feature', 'Importance', 'Selected']].to_string(index=False))

# ─── Bar chart only ───
plot_df = importance_df.sort_values('Importance', ascending=True)
colors  = ['royalblue' if f in selected_features else '#b0c4de'
           for f in plot_df['Feature']]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(plot_df['Feature'], plot_df['Importance'],
               color=colors, edgecolor='none')

for bar, val in zip(bars, plot_df['Importance']):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val}%", va='center', fontsize=10)

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor='royalblue', label='Selected'),
    Patch(facecolor='#b0c4de',   label='Not selected'),
], loc='lower right', framealpha=0.7)

ax.set_xlabel('Feature Importance (%)')
ax.set_title('Feature Importance from Random Forest Classifier', pad=12)
ax.set_facecolor('#f0f0f0')
ax.set_xlim(0, importance_df['Importance'].max() + 6)
ax.grid(axis='x', color='white', linewidth=1.5)
ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
ax.tick_params(left=False)
fig.patch.set_facecolor('#f0f0f0')

plt.tight_layout()
plt.savefig('feature_importance_rf.png', dpi=300, bbox_inches='tight')
print("\n✅ feature_importance_rf.png saved")
plt.close()