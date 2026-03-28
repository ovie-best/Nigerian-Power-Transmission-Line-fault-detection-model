import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize, MinMaxScaler
from sklearn.model_selection import learning_curve
from itertools import cycle

# ─── Reproducibility ────
np.random.seed(42)

print("=" * 70)
print("MODEL TRAINING & EVALUATION")
print("Random Forest Classifier for Nigerian Transmission Line Fault Detection")
print("=" * 70)

# ─── 1. Load Training Data ──
train_df = pd.read_csv("train_dataset.csv")
print(f"Training data loaded: {len(train_df):,} samples (Jan–Mar 2025)")

# ─── 2. Define Features and Target ───
features = ['Ia', 'Ib', 'Ic', 'Va']
target = 'Fault_Type'

X = train_df[features]
y = train_df[target]

# ─── 3. Train / Validation Split ────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set : {len(X_train):,} samples")
print(f"Validation set: {len(X_val):,} samples")

# ─── 4. Data Normalization (Min-Max Scaling) ─────
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_scaled       = scaler.transform(X)   

joblib.dump(scaler, "scaler.joblib")
print("✅ Scaler fitted on training data and saved as 'scaler.joblib'")

# ─── 5. Hyperparameter Tuning with GridSearchCV ──────
print("\n⏳ Running GridSearchCV (this may take a while)...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
grid_search.fit(X_train_scaled, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best CV accuracy:", grid_search.best_score_)

# Use the best model
model = grid_search.best_estimator_
print("\n✅ Model training completed (best GridSearchCV estimator on 4 scaled features)")

joblib.dump(model, "model.joblib")
print("✅ Model saved as 'model.joblib'")

# ─── 6. Evaluate on Validation Set ──────
y_pred = model.predict(X_val_scaled)
accuracy = accuracy_score(y_val, y_pred)
print(f"\nModel Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_val, y_pred, digits=4))

# ─── 7. Confusion Matrix ─────────────
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

# ─── 8. ENHANCED EVALUATION ──────────
print("\n" + "="*70)
print("ENHANCED EVALUATION – ROC, METRICS TABLE & LEARNING CURVE")
print("="*70)

# 8.1 ROC Curve — subsample validation set for speed
print("\n=== ROC Curve Analysis (4 selected features) ===")
MAX_ROC_SAMPLES = 5000
if len(X_val_scaled) > MAX_ROC_SAMPLES:
    roc_idx = np.random.choice(len(X_val_scaled), MAX_ROC_SAMPLES, replace=False)
    X_roc = X_val_scaled[roc_idx]
    y_roc = y_val.iloc[roc_idx]
else:
    X_roc, y_roc = X_val_scaled, y_val

y_val_bin = label_binarize(y_roc, classes=[0, 1, 2, 3, 4, 5])
n_classes = y_val_bin.shape[1]
y_score = model.predict_proba(X_roc)

fpr, tpr, roc_auc = {}, {}, {}
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
plt.title('ROC Curve Analysis – One-vs-Rest\n(4 Selected Features: Ia, Ib, Ic, Va)')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curve_analysis_4features.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ ROC curve saved")

# 8.2 Performance Metrics Table
print("\n=== Performance Metrics Table (4 features) ===")
report = classification_report(y_val, y_pred, output_dict=True, digits=4)
metrics_df = pd.DataFrame(report).transpose().round(4)
print(metrics_df.to_string())
metrics_df.to_csv("performance_metrics_table_4features.csv", index=True)
metrics_df.to_latex("performance_metrics_table_4features.tex", float_format="%.4f")
print("✅ Metrics table saved")

# 8.3 Learning Curve
print("\n=== Learning Curve Analysis ===")

lc_model = RandomForestClassifier(
    **grid_search.best_params_,   
    random_state=42,
    n_jobs=-1
)

train_sizes, train_scores, val_scores = learning_curve(
    lc_model,
    X_train_scaled, y_train,      
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

train_mean = np.mean(train_scores, axis=1)
train_std  = np.std(train_scores, axis=1)
val_mean   = np.mean(val_scores, axis=1)
val_std    = np.std(val_scores, axis=1)

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
print("✅ Learning curve saved")

print("\n✅ Training & evaluation completed successfully!")