"""
Title: train_baseline_models.py

Purpose:
    Train basic supervised machine learning classifiers to 
    predict free throw outcomes using kinematic joint angles 
    at the moment of release.

Input:
    - X_release.csv (features)
    - y_release.csv (labels)

Output:
    - Classification report printed to console
    - Accuracy scores of basic models


ML Notes:
- Precision: How many of the predicted "made" shots were actually made?
- Recall: How many of the actual "made" shots were correctly predicted? 
- F1 Score: balance between precision and recall, useful for imbalanced classes
- Support: Count of occurrences of each class in the test set
    
    
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB


# ========================================
# Parameters
# ========================================

test_percent = 0.2 # Percentage of data to use for testing (remainder used for training)

# ========================================
# Paths
# ========================================

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parents[2]
data_dir = base_dir / "data"
release_dir = data_dir / "summary" / "release"

X_path = release_dir / "X_release.csv"
y_path = release_dir / "y_release.csv"

# ========================================
# Load Prepared Features and Labels
# ========================================
print(f"Reading input from: {X_path}")
print(f"Reading input from: {X_path}")
X = pd.read_csv(X_path)
y = pd.read_csv(y_path).squeeze()  # convert single-column DataFrame to Series

print(f"X shape: {X.shape}")
print(f"y distribution:\n{y.value_counts()}")


# ========================================
# Train/Test Split
# ========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_percent, random_state=42, stratify=y
)

# ========================================
# Feature Scaling
# ========================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================================
# Train and Evaluate Models
# ========================================

models = {
    "Logistic Regression": LogisticRegression(),                      # Linear model; best for simple, linearly separable data
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),       # Classifies based on the majority vote of closest neighbors
    "Support Vector Machine": SVC(kernel='rbf', probability=True),    # Finds the optimal boundary (hyperplane) with maximum margin
    "Decision Tree": DecisionTreeClassifier(),                        # Splits data using feature thresholds to form a tree of decisions
    "Random Forest": RandomForestClassifier(n_estimators=100),        # Combines many decision trees to reduce overfitting
    "Gradient Boosting": GradientBoostingClassifier(),                # Builds trees in sequence, each correcting the last
    "Naive Bayes": GaussianNB()                                       # Uses probability based on Bayes' Theorem; assumes feature independence
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred, target_names=["miss", "made"]))
