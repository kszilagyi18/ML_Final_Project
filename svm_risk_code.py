import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -------------------------------
# CREATE OUTPUT DIRECTORY
# -------------------------------
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# LOAD DATASET
# -------------------------------

df = pd.read_csv(r"insurance\medical_insurance.csv")
print("Loaded dataset shape:", df.shape)

# ---------------------------------------
# 1. DEFINE TARGET AND FEATURES
# ---------------------------------------
y = df["is_high_risk"]   # TARGET COLUMN


#drop missing participants
df.drop(columns=['person_id'], inplace=True)
df = df[df['alcohol_freq'].notnull()].copy()

y = df["is_high_risk"]   # TARGET COLUMN

# Drop categories that aren't related to demographics or health
df = df.drop(columns=['age','is_high_risk','risk_score','annual_medical_cost', 'annual_premium', 'monthly_premium', 'claims_count', 'avg_claim_amount', 'total_claims_paid', 'plan_type', 'network_tier', 'deductible',
                      'copay', 'policy_term_years', 'policy_changes_last_2yrs', 'provider_quality'])

X = pd.get_dummies(X, drop_first=True)
print("Feature matrix shape:", X.shape)

# ---------------------------------------
# 2. TRAIN / TEST SPLIT
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------
# 3. SCALING
# ---------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------
# 4. HYPERPARAMETER SEARCH (SVM)
# ---------------------------------------
param_dist = {
    "C": [0.01, 0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

svm = SVC(class_weight="balanced")

search = RandomizedSearchCV(
    svm,
    param_distributions=param_dist,
    n_iter=10,
    scoring="f1",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train_scaled, y_train)
best_svm = search.best_estimator_

print("Best parameters:", search.best_params_)

# ---------------------------------------
# 5. PREDICTIONS + METRICS
# ---------------------------------------
y_pred = best_svm.predict(X_test_scaled)

print("\n--- CLASSIFICATION REPORT ---\n")
print(classification_report(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"F1 Score : {f1:.3f}")

# ---------------------------------------
# 6. CONFUSION MATRIX
# ---------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted Low Risk", "Predicted High Risk"],
    yticklabels=["Actual Low Risk", "Actual High Risk"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix (Target = is_high_risk)")
plt.tight_layout()

# Save the confusion matrix plot
confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
print(f"\nConfusion matrix saved to: {confusion_matrix_path}")

plt.show()

# ---------------------------------------
# 7. METRICS BAR CHART (OPTIONAL)
# ---------------------------------------
metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Score': [acc, prec, rec, f1]
}

plt.figure(figsize=(8,6))
plt.bar(metrics_data['Metric'], metrics_data['Score'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('SVM Model Performance Metrics')
plt.grid(axis='y', alpha=0.3)

for i, v in enumerate(metrics_data['Score']):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

# Save the metrics plot
metrics_path = os.path.join(output_dir, "performance_metrics.png")
plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
print(f"Performance metrics saved to: {metrics_path}")

plt.show()