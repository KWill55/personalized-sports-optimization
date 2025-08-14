import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# ========================================
# Parameters
# ========================================
test_percent = 0.2  # % of data used for testing
show_classification_reports = False  # Set to True to see full reports

# ========================================
# Paths
# ========================================
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parents[1]
summary_dir = base_dir / "data" / "tests" / "player_tracking_tests" / "summary"
X_path = summary_dir / "X_release.csv"
y_path = summary_dir / "y_release.csv"

# ========================================
# Load Data
# ========================================
print(f"Reading input from: {X_path}")
X = pd.read_csv(X_path)
y = pd.read_csv(y_path).squeeze()

print(f"X shape: {X.shape}")
print(f"y distribution:\n{y.value_counts()}")

# ========================================
# Split & Scale
# ========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_percent, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================================
# Train Models
# ========================================
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Support Vector Machine": SVC(kernel='rbf', probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Naive Bayes": GaussianNB()
}

print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")

# Collect results
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision (made)": prec,
        "Recall (made)": rec,
        "F1 Score (made)": f1
    })

    if show_classification_reports:
        print(f"\n{name} - Full Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["miss", "made"]))

# ========================================
# Display Results Table
# ========================================
results_df = pd.DataFrame(results)
print("\n=== Summary Table ===")
print(results_df.round(3).sort_values(by="F1 Score (made)", ascending=False).to_string(index=False))
