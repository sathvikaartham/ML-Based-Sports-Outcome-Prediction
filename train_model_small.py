
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score

# =============================================================
# PATHS AND CONFIGURATION
# =============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"
DATA_PATH = BASE_DIR / "Dataset" / "ball_by_ball_it20.csv"

# IMPORTANT: Never overwrite the original model.
OUTPUT_PATH = MODEL_DIR / "models_small_candidate.joblib"

# Smaller model settings to reduce runtime memory.
N_TREES = 30
MAX_DEPTH = 12
MIN_SAMPLES_LEAF = 5

RANDOM_STATE = 42

# =============================================================
# FEATURES AND TARGETS
# =============================================================

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
    "Match ID",
]

# =============================================================
# LOAD DATASET
# =============================================================

print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    usecols=FEATURES + TARGET_COLUMNS + ["Winner"]
)

print("Dataset loaded:", len(df), "rows")

# =============================================================
# PREPARE FEATURES
# =============================================================

X = df[FEATURES].apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)

# =============================================================
# PREPARE CLASSIFICATION TARGET
# =============================================================

y_class = pd.to_numeric(
    df["Chased Successfully"],
    errors="coerce"
)

# Fallback if the classification target is missing.
missing_class = y_class.isna()

if missing_class.any():
    fallback = (
        df["Winner"].astype(str).str.strip()
        == df["Bat Second"].astype(str).str.strip()
    ).astype(int)

    y_class.loc[missing_class] = fallback.loc[missing_class]

# =============================================================
# PREPARE SCORE TARGET
# =============================================================

y_score = pd.to_numeric(
    df["Target Score"],
    errors="coerce"
)

# =============================================================
# FILTER INVALID ROWS
# =============================================================

match_ids = df["Match ID"]

valid = (
    y_class.notna()
    & y_score.notna()
    & match_ids.notna()
    & np.isfinite(y_class)
    & np.isfinite(y_score)
)

X = X.loc[valid].reset_index(drop=True)

y_class = (
    y_class.loc[valid]
    .astype(int)
    .reset_index(drop=True)
)

y_score = (
    y_score.loc[valid]
    .astype(float)
    .reset_index(drop=True)
)

match_ids = (
    match_ids.loc[valid]
    .reset_index(drop=True)
)

if not y_class.isin([0, 1]).all():
    raise ValueError(
        "Classification target must contain only 0 and 1."
    )

if y_class.nunique() != 2:
    raise ValueError(
        "Both classes (0 and 1) must be present."
    )

if match_ids.nunique() < 2:
    raise ValueError(
        "At least two unique matches are required."
    )

print("Valid training rows:", len(X))
print("Unique matches:", match_ids.nunique())

# =============================================================
# PREPARE TEAM LIST
# =============================================================

teams = pd.concat(
    [df["Bat First"], df["Bat Second"]],
    ignore_index=True
).dropna().astype(str).str.strip()

teams = sorted(
    team
    for team in teams.unique().tolist()
    if team
)

print("Teams:", len(teams))

# =============================================================
# MATCH-BASED TRAIN / TEST SPLIT
# =============================================================

print("\nCreating match-based evaluation split...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y_class,
        groups=match_ids
    )
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y_class.iloc[train_idx]
y_test = y_class.iloc[test_idx]

y_score_train = y_score.iloc[train_idx]
y_score_test = y_score.iloc[test_idx]

train_matches = set(match_ids.iloc[train_idx])
test_matches = set(match_ids.iloc[test_idx])

print("Training rows:", len(train_idx))
print("Testing rows:", len(test_idx))
print("Training matches:", len(train_matches))
print("Testing matches:", len(test_matches))

# Verify no match is shared between train and test.
if train_matches.intersection(test_matches):
    raise RuntimeError(
        "Data leakage detected: a match appears in both sets."
    )

if y_train.nunique() != 2 or y_test.nunique() != 2:
    raise ValueError(
        "The match-based split must contain both classes "
        "in training and testing. No model was saved."
    )

# =============================================================
# EVALUATE SMALL CLASSIFIER
# =============================================================

print("\nEvaluating smaller classifier...")

evaluation_scaler = MinMaxScaler()

X_train_scaled = evaluation_scaler.fit_transform(X_train)
X_test_scaled = evaluation_scaler.transform(X_test)

evaluation_model = RandomForestClassifier(
    n_estimators=N_TREES,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=1,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF
)

evaluation_model.fit(
    X_train_scaled,
    y_train
)

predictions = evaluation_model.predict(
    X_test_scaled
)

accuracy = round(
    accuracy_score(y_test, predictions) * 100,
    2
)

print(
    "Match-based held-out accuracy:",
    accuracy,
    "%"
)

del evaluation_model
del evaluation_scaler
del X_train_scaled
del X_test_scaled
del predictions

gc.collect()

# =============================================================
# TRAIN FINAL SMALL MODELS ON ALL VALID DATA
# =============================================================

print("\nTraining final smaller models...")

# Feature scaler.
feature_scaler = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(X)

# Score scaler.
score_scaler = MinMaxScaler()

score_scaled = score_scaler.fit_transform(
    y_score.to_numpy().reshape(-1, 1)
).ravel()

# =============================================================
# FINAL CLASSIFIER
# =============================================================

print("Training classifier...")

classifier = RandomForestClassifier(
    n_estimators=N_TREES,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=1,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF
)

classifier.fit(
    X_scaled,
    y_class
)

# =============================================================
# FINAL REGRESSOR
# =============================================================

print("Training regressor...")

regressor = RandomForestRegressor(
    n_estimators=N_TREES,
    random_state=RANDOM_STATE,
    n_jobs=1,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF
)

regressor.fit(
    X_scaled,
    score_scaled
)

# =============================================================
# SAVE CANDIDATE MODEL
# =============================================================

artifacts = {
    "classifier": classifier,
    "regressor": regressor,
    "feature_scaler": feature_scaler,
    "score_scaler": score_scaler,
    "teams": teams,
    "accuracy": accuracy,
    "model_name": "Small Random Forest",
    "training_source": "Dataset/ball_by_ball_it20.csv",
}

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("\nSaving candidate model...")

joblib.dump(
    artifacts,
    OUTPUT_PATH,
    compress=3
)

print("\nTraining complete.")
print("Match-based accuracy:", accuracy, "%")
print("Classifier trees:", N_TREES)
print("Regressor trees:", N_TREES)
print("Maximum depth:", MAX_DEPTH)
print("Minimum samples per leaf:", MIN_SAMPLES_LEAF)
print("Teams:", len(teams))
print("Saved candidate:", OUTPUT_PATH)
print(
    "Candidate size:",
    round(OUTPUT_PATH.stat().st_size / 1024**2, 2),
    "MiB"
)

# =============================================================
# CLEAN UP
# =============================================================

del df
del X, X_scaled
del y_class, y_score, score_scaled
del classifier, regressor, artifacts
del train_idx, test_idx
del X_train, X_test
del y_train, y_test
del y_score_train, y_score_test
del match_ids

gc.collect()

print("Done.")