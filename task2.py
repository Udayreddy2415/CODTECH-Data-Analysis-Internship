# ===========================================
# Predictive Analysis using Machine Learning
# Task: Build a model to predict outcomes
# Dataset: Breast Cancer (binary classification)
# Deliverable: Demonstration of feature selection,
#              model training, and evaluation
# ===========================================

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, RocCurveDisplay
)

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset Shape:", X.shape)
print("Target Classes:", data.target_names)

# -------------------------------
# Step 2: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Step 3: Feature Selection
# Use ANOVA F-test (SelectKBest) to choose top features
# -------------------------------
selector = SelectKBest(score_func=f_classif, k=10)  # pick top 10 features
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = X.columns[selector.get_support()]
print("Selected Features:", list(selected_features))

# -------------------------------
# Step 4: Standardize Features
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# -------------------------------
# Step 5: Train Models
# Logistic Regression + Random Forest
# -------------------------------
log_reg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

log_reg.fit(X_train_scaled, y_train)
rf.fit(X_train_selected, y_train)

# -------------------------------
# Step 6: Evaluate Models
# -------------------------------
models = {
    "Logistic Regression": log_reg,
    "Random Forest": rf
}

for name, model in models.items():
    if name == "Logistic Regression":
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred = model.predict(X_test_selected)
        y_proba = model.predict_proba(X_test_selected)[:, 1]
    
    print(f"\\n=== {name} Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("Classification Report:\\n", classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title(f"ROC Curve - {name}")
    plt.show()
    
