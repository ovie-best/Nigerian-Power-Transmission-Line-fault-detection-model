import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime, timedelta

# ─── NEW IMPORTS FOR ROC, METRICS TABLE & LEARNING CURVE ─────────────────────
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve
from itertools import cycle

# ─── Reproducibility ────────────────────────────────────────────────────────
np.random.seed(42)

print("=" * 70)
print("MODEL TRAINING & PREDICTION")
print("Random Forest Classifier for Nigerian Transmission Line Fault Detection")
print("=" * 70)

# ─── 1. Load Training Data ───────────────────────────────────────────────────
train_df = pd.read_csv("train_dataset.csv")
print(f"Training data loaded: {len(train_df):,} samples (Jan–Mar 2025)")

# ─── 2. Define Features and Target ───────────────────────────────────────────
# ONLY the 4 selected features (Ia, Ib, Ic, Va) as per Section 3.3.3 & Tables 3.1–3.2
features = ['Ia', 'Ib', 'Ic', 'Va']
target = 'Fault_Type'

X = train_df[features]
y = train_df[target]

# ─── 3. Train / Validation Split (80/20) ─────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set : {len(X_train):,} samples")
print(f"Validation set: {len(X_val):,} samples")

# ─── 4. Train Random Forest Classifier ───────────────────────────────────────
# Using the tuned hyperparameters from Section 3.4.1 (GridSearchCV)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("\n✅ Model training completed (tuned Random Forest on 4 features)")

# ─── 5. Evaluate on Validation Set ───────────────────────────────────────────
y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print(f"\nModel Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_val, y_pred, digits=4))

# ─── 6. Confusion Matrix (saved for thesis) ──────────────────────────────────
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[0,1,2,3,4,5], yticklabels=[0,1,2,3,4,5])
plt.title('Fault Type Confusion Matrix (Validation Set)')
plt.xlabel('Predicted Fault Type')
plt.ylabel('Actual Fault Type')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# ─── 7. NEW ENHANCED EVALUATION (ROC + Metrics Table + Learning Curve) ────────
# This block uses ONLY the 4 selected features and the tuned model

print("\n" + "="*70)
print("ENHANCED EVALUATION – ROC, PERFORMANCE TABLE & LEARNING CURVE")
print("="*70)

# 7.1 ROC Curve Analysis (One-vs-Rest Multiclass)
print("\n=== 7.1 ROC Curve Analysis (4 selected features) ===")
y_val_bin = label_binarize(y_val, classes=[0, 1, 2, 3, 4, 5])
n_classes = y_val_bin.shape[1]
y_score = model.predict_proba(X_val)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Class {i} (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Analysis – One-vs-Rest\n(Random Forest, 4 Selected Features: Ia, Ib, Ic, Va)')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curve_analysis_4features.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ ROC curve saved as 'roc_curve_analysis_4features.png'")

# 7.2 Performance Metrics Table (clean & publication-ready)
print("\n=== 7.2 Performance Metrics Table (4 features) ===")
report = classification_report(y_val, y_pred, output_dict=True, digits=4)
metrics_df = pd.DataFrame(report).transpose().round(4)

# Add overall accuracy row
metrics_df.loc['accuracy'] = [accuracy_score(y_val, y_pred), '', '', '', len(y_val)]

print(metrics_df.to_string())
metrics_df.to_csv("performance_metrics_table_4features.csv", index=True)
metrics_df.to_latex("performance_metrics_table_4features.tex", float_format="%.4f")
print("✅ Performance metrics table saved as 'performance_metrics_table_4features.csv' and .tex")

# 7.3 Learning Curve Analysis
print("\n=== 7.3 Learning Curve Analysis (4-feature model) ===")
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.plot(train_sizes, val_mean, 'o-', color="g", label="Cross-validation score")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="g")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy Score")
plt.title("Learning Curve Analysis\n(Random Forest – 4 Selected Features)")
plt.legend(loc="best")
plt.grid(True)
plt.savefig('learning_curve_analysis_4features.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Learning curve saved as 'learning_curve_analysis_4features.png'")

# ─── 8. Predict on Future Test Set (Apr–Jun 2025) ───────────────────────────
test_df = pd.read_csv("test_dataset.csv")
print(f"\nTest data loaded: {len(test_df):,} samples (Apr–Jun 2025)")

# Add realistic 15-minute timestamps (matches generator logic)
test_start = datetime(2025, 4, 1, 0, 0, 0)
test_df["Timestamp"] = [test_start + timedelta(minutes=15 * i) 
                        for i in range(len(test_df))]

# Predict using ONLY the 4 selected features
X_test = test_df[features]
test_df["Predicted_Fault_Type"] = model.predict(X_test)

# ─── 9. Save Predictions for Streamlit Dashboard ─────────────────────────────
test_df.to_csv("nigerian_test_data_with_predictions.csv", index=False)
print("\n✅ Predictions saved to 'nigerian_test_data_with_predictions.csv'")

# ─── 10. Quick Summary of Predictions ────────────────────────────────────────
print("\n─── Predicted Fault-Type Distribution (Apr–Jun 2025) ───")
pred_dist = test_df['Predicted_Fault_Type'].value_counts().sort_index()
label_map = {0: 'Class 0 – Normal A', 1: 'Class 1 – Normal B',
             2: 'Class 2 – LG fault', 3: 'Class 3 – LL fault',
             4: 'Class 4 – LLG fault', 5: 'Class 5 – Healthy'}

for cls, count in pred_dist.items():
    pct = 100 * count / len(test_df)
    print(f" {label_map.get(cls, cls):28s} {count:5,} ({pct:5.1f} %)")

print(f"\nTotal predictions generated: {len(test_df):,}")