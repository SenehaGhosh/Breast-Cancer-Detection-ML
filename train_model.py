import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

print("1. Loading Wisconsin Breast Cancer dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0 = Malignant, 1 = Benign (Note: sklearn standard maps 0=malignant, 1=benign)

# Train-test split (stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("2. Applying StandardScaler (preventing data leakage)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models to benchmark
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "SVM (RBF Kernel)": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

print("\n3. Training and evaluating models...")
best_model_name = None
best_score = 0
best_model = None

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    # Using ROC-AUC as comparative metric
    auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    print(f" -> {name} | ROC-AUC: {auc:.4f}")
    
    if auc > best_score:
        best_score = auc
        best_model_name = name
        best_model = model

print(f"\n🏆 Best Performing Model: {best_model_name} (ROC-AUC: {best_score:.4f})")

# Hyperparameter tuning on the best model (using SVM as example or dynamic)
print("4. Tuning hyperparameters...")
if "SVM" in best_model_name:
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1]}
    grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=5, scoring='roc_auc')
    grid.fit(X_train_scaled, y_train)
    best_model = grid.best_estimator_
    print(f"Tuned SVM Parameters: {grid.best_params_}")
elif "Random Forest" in best_model_name:
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='roc_auc')
    grid.fit(X_train_scaled, y_train)
    best_model = grid.best_estimator_

# Save the trained model and the scaler bundle
print("5. Saving model and scaler artifacts...")
joblib.dump(best_model, "breast_cancer_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Training complete! Saved 'breast_cancer_model.pkl' and 'scaler.pkl'.")