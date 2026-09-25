
from pathlib import Path
import gc
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

DATA_PATH = BASE_DIR / "Dataset" / "ball_by_ball_it20.csv"

FEATURES = [
    "Runs From Ball",
    "Innings Runs",
    "Innings Wickets",
    "Balls Remaining",
    "Total Batter Runs",
    "Total Non Striker Runs",
    "Batter Balls Faced",
    "Non Striker Balls Faced",
]

TARGET_COLUMNS = [
    "Chased Successfully",
    "Target Score",
    "Bat First",
    "Bat Second",
]

print("Loading dataset...")

# Load only the columns required by the models.
df = pd.read_csv(
    DATA_PATH,
    usecols=FEATURES + TARGET_COLUMNS + ["Winner"]
)

print("Dataset loaded:", len(df), "rows")

# Prepare features.
X = df[FEATURES].apply(
    pd.to_numeric,
    errors="coerce"
)
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# Prepare classification target.
if "Chased Successfully" in df.columns:
    y_class = pd.to_numeric(
        df["Chased Successfully"],
        errors="coerce"
    )
else:
    y_class = (
        df["Winner"].astype(str).str.strip()
        == df["Bat Second"].astype(str).str.strip()
    ).astype(int)

# Prepare score target.
y_score = pd.to_numeric(
    df["Target Score"],
    errors="coerce"
)

valid = (
    y_class.notna()
    & y_score.notna()
    & np.isfinite(y_class)
    & np.isfinite(y_score)
)

X = X.loc[valid].reset_index(drop=True)
y_class = y_class.loc[valid].astype(int).reset_index(drop=True)
y_score = y_score.loc[valid].astype(float).reset_index(drop=True)

if not y_class.isin([0, 1]).all():
    raise ValueError("Classification target must contain only 0 and 1.")

print("Valid training rows:", len(X))

# Save the team list.
teams = pd.concat(
    [df["Bat First"], df["Bat Second"]],
    ignore_index=True
).dropna().astype(str).str.strip()

teams = sorted(
    team for team in teams.unique().tolist()
    if team
)

# Calculate a held-out accuracy.
accuracy = None

if len(X) >= 10 and y_class.nunique() == 2 and y_class.value_counts().min() >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_class,
        test_size=0.20,
        random_state=42,
        stratify=y_class
    )

    evaluation_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
        max_depth=16,
        min_samples_leaf=2
    )

    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)

    accuracy = round(
        accuracy_score(y_test, predictions) * 100,
        2
    )

    del evaluation_model
    del X_train, X_test, y_train, y_test, predictions
    gc.collect()

print("Training final models...")

# Fit the feature scaler.
feature_scaler = MinMaxScaler()
X_scaled = feature_scaler.fit_transform(X)

# Fit the score scaler.
score_scaler = MinMaxScaler()
score_scaled = score_scaler.fit_transform(
    y_score.to_numpy().reshape(-1, 1)
).ravel()

# Train final classifier.
classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=1,
    max_depth=16,
    min_samples_leaf=2
)
classifier.fit(X_scaled, y_class)

# Train final regressor.
regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=1,
    max_depth=16,
    min_samples_leaf=2
)
regressor.fit(X_scaled, score_scaled)

# Save all artifacts together.
artifacts = {
    "classifier": classifier,
    "regressor": regressor,
    "feature_scaler": feature_scaler,
    "score_scaler": score_scaler,
    "teams": teams,
    "accuracy": accuracy,
    "model_name": "Random Forest Classifier",
    "training_source": "Dataset/ball_by_ball_it20.csv",
}

output_path = MODEL_DIR / "models.joblib"
joblib.dump(artifacts, output_path, compress=3)

print("\nTraining complete.")
print("Accuracy:", accuracy)
print("Teams:", len(teams))
print("Saved:", output_path)

# Clean up.
del df, X, X_scaled, y_class, y_score, score_scaled
del classifier, regressor
gc.collect()