
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load previously prepared features
data = pd.read_csv("outputs/lapse_features.csv")

X = data.drop(columns=["customer_id", "lapsed"])
y = data["lapsed"]

# Numeric and categorical features
numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "average_discount",
    "cancellation_rate",
    "average_delivery_minutes",
    "age",
    "tenure_days"
]

categorical_features = [
    "favourite_category",
    "city",
    "gender"
]

# Split before fitting any scaler or encoder
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Training-only preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# Baseline: predict everyone as not lapsed
baseline_predictions = np.zeros(len(y_test), dtype=int)

print("\nBASELINE MODEL")
print("Accuracy:", round(accuracy_score(y_test, baseline_predictions), 4))
print("F1:", round(f1_score(y_test, baseline_predictions, zero_division=0), 4))

# Logistic Regression pipeline
logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

logistic_grid = GridSearchCV(
    logistic_pipeline,
    param_grid={
        "model__C": [0.01, 0.1, 1, 10, 100]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("\nTraining Logistic Regression...")
logistic_grid.fit(X_train, y_train)

print("Best parameters:", logistic_grid.best_params_)
print("Best CV F1:", round(logistic_grid.best_score_, 4))

# Random Forest pipeline
forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )
])

forest_grid = GridSearchCV(
    forest_pipeline,
    param_grid={
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("\nTraining Random Forest...")
forest_grid.fit(X_train, y_train)

print("Best parameters:", forest_grid.best_params_)
print("Best CV F1:", round(forest_grid.best_score_, 4))

# Evaluate only after both grid searches are complete
def evaluate(name, model):
    predictions = model.predict(X_test)

    print(f"\n{name} TEST RESULTS")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
    print("Recall:", round(recall_score(y_test, predictions, zero_division=0), 4))
    print("F1:", round(f1_score(y_test, predictions, zero_division=0), 4))

    return f1_score(y_test, predictions, zero_division=0)

logistic_f1 = evaluate(
    "Logistic Regression",
    logistic_grid.best_estimator_
)

forest_f1 = evaluate(
    "Random Forest",
    forest_grid.best_estimator_
)

print("\nMODEL COMPARISON")
print("Logistic Regression F1:", round(logistic_f1, 4))
print("Random Forest F1:", round(forest_f1, 4))

# Confirm that the scaler learned only from training data
trained_scaler = logistic_grid.best_estimator_.named_steps[
    "preprocessor"
].named_transformers_["num"]

frequency_index = numeric_features.index("frequency")

print("\nTraining frequency mean:",
      round(trained_scaler.mean_[frequency_index], 4))
print("Full-cohort frequency mean:",
      round(X["frequency"].mean(), 4))
print("The means differ because the scaler was fitted only on training rows.")

print("\nMODEL TRAINING COMPLETED")


# PART B FINAL: Class balance, feature importance, model saving
import joblib
from pathlib import Path
from sklearn.base import clone

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

# 1. Balance only the training rows
train_data = X_train.copy()
train_data["lapsed"] = y_train

majority = train_data[train_data["lapsed"] == 0]
minority = train_data[train_data["lapsed"] == 1]

extra_minority = minority.sample(
    n=len(majority) - len(minority),
    replace=True,
    random_state=42
)

balanced = pd.concat(
    [train_data, extra_minority],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

X_balanced = balanced.drop(columns=["lapsed"])
y_balanced = balanced["lapsed"]

print("\nBALANCED TRAINING CLASS COUNTS")
print(y_balanced.value_counts().sort_index())

# 2. Refit best logistic regression on balanced training data
balanced_logistic = clone(logistic_grid.best_estimator_)
balanced_logistic.fit(X_balanced, y_balanced)

def show_metrics(name, model):
    pred = model.predict(X_test)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    print(f"\n{name}")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    return precision, recall, f1

original_metrics = show_metrics(
    "Original Logistic Regression",
    logistic_grid.best_estimator_
)

balanced_metrics = show_metrics(
    "Balanced Logistic Regression",
    balanced_logistic
)

metric_names = ["precision", "recall", "F1"]
differences = [
    abs(b - a)
    for a, b in zip(original_metrics, balanced_metrics)
]
print(
    "Metric with largest change:",
    metric_names[int(np.argmax(differences))]
)

# 3. Random Forest top five important features
forest_best = forest_grid.best_estimator_
forest_feature_names = forest_best.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = pd.Series(
    forest_best.named_steps["model"].feature_importances_,
    index=forest_feature_names
).sort_values(ascending=False)

print("\nTOP FIVE RANDOM FOREST FEATURES")
print(importance.head(5))

# 4. Logistic Regression coefficient signs
logistic_best = logistic_grid.best_estimator_
logistic_feature_names = logistic_best.named_steps[
    "preprocessor"
].get_feature_names_out()

coefficients = pd.Series(
    logistic_best.named_steps["model"].coef_[0],
    index=logistic_feature_names
)

print("\nLOGISTIC REGRESSION COEFFICIENTS")
for feature in ["num__recency", "num__frequency", "num__monetary"]:
    value = coefficients[feature]
    sign = "positive" if value > 0 else "negative"
    print(f"{feature}: {value:.4f} ({sign})")

# 5. Save selected model
selected_model = logistic_best
model_path = OUTPUTS / "lapse_model.joblib"

joblib.dump(selected_model, model_path)

# Reload and confirm predictions match
reloaded_model = joblib.load(model_path)

original_predictions = selected_model.predict(X_test)
reloaded_predictions = reloaded_model.predict(X_test)

assert np.array_equal(
    original_predictions,
    reloaded_predictions
), "Reloaded model predictions do not match!"

print("\nModel saved and reloaded successfully!")

# 6. Create Premium retention list
segments = pd.read_csv(OUTPUTS / "customer_segments.csv")

# X_test retains the original feature-table row indices
test_customers = data.loc[
    X_test.index, ["customer_id", "lapsed"]
].copy()

test_customers["lapse_probability"] = (
    selected_model.predict_proba(X_test)[:, 1]
)

test_customers = test_customers.merge(
    segments[["customer_id", "is_premium"]],
    on="customer_id",
    how="left"
)

premium_test = test_customers[
    test_customers["is_premium"].fillna(False).astype(bool)
].copy()

retention = premium_test.sort_values(
    "lapse_probability",
    ascending=False
).head(10)

retention = retention.rename(
    columns={"lapsed": "actually_lapsed"}
)

retention = retention[
    ["customer_id", "lapse_probability", "actually_lapsed"]
]

retention.to_csv(
    OUTPUTS / "retention_list.csv",
    index=False
)

print("\nRETENTION LIST")
print(retention.to_string(index=False))
print("Retention customers:", len(retention))
print("Actually lapsed:", int(retention["actually_lapsed"].sum()))

print("\nPART B MODEL OUTPUTS SAVED")
print("outputs/lapse_model.joblib")
print("outputs/retention_list.csv")
