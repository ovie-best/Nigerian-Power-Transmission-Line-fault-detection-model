import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

# Load and prepare data (same as before)
df = pd.read_csv('synthetic_fault_data_2017_2025.csv')

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

# Time-based split
train_mask = df['Year'] <= 2023
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

# Train model
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# 1. Dynamically get the classes the model actually saw during training
actual_classes = model.classes_
n_classes = len(actual_classes)

# 2. Binarize labels using ONLY the actual classes
y_test_bin = label_binarize(y_test, classes=actual_classes)
y_score = model.predict_proba(X_test)

# 3. Handle binary classification edge case 
# (label_binarize returns a 1D array if there are only 2 classes, which breaks the 2D loop)
if n_classes == 2:
    y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

# Plot ROC curves
plt.figure(figsize=(10, 8))
# Use a dynamic colormap to handle whatever number of classes the model finds
colors = plt.cm.get_cmap('tab10', n_classes)

for i in range(n_classes):
    class_label = actual_classes[i]
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors(i), lw=2,
             label=f'Class {class_label} (AUC = {roc_auc:.3f})')

# Micro-average ROC
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, color='deeppink', linestyle='--', lw=3,
         label=f'Micro-average ROC (AUC = {roc_auc_micro:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Multi-Class ROC Curves (Found {n_classes} Classes in Training)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ROC_Curves_MultiClass.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ ROC_Curves_MultiClass.png saved")