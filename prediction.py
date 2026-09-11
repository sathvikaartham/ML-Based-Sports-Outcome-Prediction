from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent

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

class Predictor:
    """
    Web-deployment version of the Random Forest parts of the original
    MatchResult notebook.

    The original notebook:
      - uses these 8 match-state features,
      - MinMax-scales them to [0, 1],
      - trains RandomForestRegressor for score forecasting,
      - trains RandomForestClassifier for the winner/chase class.

    If Dataset/ball_by_ball_it20.csv is present, up to the first 30,000
    records are used, matching the notebook. If that dataset is not present,
    the app falls back to testData.csv so the demo can still run.
    """

    def __init__(self):
        self.teams = []
        self.training_source = ""
        self.feature_scaler = MinMaxScaler((0, 1))
        self.score_scaler = MinMaxScaler((0, 1))

        dataset_path = BASE_DIR / "Dataset" / "ball_by_ball_it20.csv"
        test_path = BASE_DIR / "testData.csv"
        root_test_path = BASE_DIR / "Dataset" / "testData.csv"

        if dataset_path.exists():
            df = pd.read_csv(dataset_path, nrows=30000)
            self.training_source = "Dataset/ball_by_ball_it20.csv (up to 30,000 rows)"
            self._train_from_full_dataset(df)
        elif test_path.exists():
            df = pd.read_csv(test_path)
            self.training_source = "testData.csv fallback (demo mode)"
            self._train_from_test_data(df)
        elif root_test_path.exists():
            df = pd.read_csv(root_test_path)
            self.training_source = "Dataset/testData.csv fallback (demo mode)"
            self._train_from_test_data(df)
        else:
            raise FileNotFoundError(
                "No training data found. Add Dataset/ball_by_ball_it20.csv "
                "or testData.csv to the project."
            )

    def _prepare_features(self, df):
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError("Missing feature columns: " + ", ".join(missing))

        X = df[FEATURES].copy().fillna(0)
        return X.astype(float)

    def _train_from_full_dataset(self, df):
        required = FEATURES + ["Chased Successfully", "Target Score", "Bat First", "Bat Second"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError("Dataset is missing columns: " + ", ".join(missing))

        X_raw = self._prepare_features(df)
        y_class = pd.to_numeric(df["Chased Successfully"], errors="coerce").fillna(0).astype(int)
        y_score = pd.to_numeric(df["Target Score"], errors="coerce").fillna(0).astype(float)

        self.feature_scaler.fit(X_raw)
        X = self.feature_scaler.transform(X_raw)

        self.score_scaler.fit(y_score.to_numpy().reshape(-1, 1))
        y_score_scaled = self.score_scaler.transform(
            y_score.to_numpy().reshape(-1, 1)
        ).ravel()

        self.classifier = RandomForestClassifier(random_state=42)
        self.regressor = RandomForestRegressor(random_state=42)
        self.classifier.fit(X, y_class)
        self.regressor.fit(X, y_score_scaled)

        teams = pd.concat([df["Bat First"], df["Bat Second"]], ignore_index=True)
        self.teams = sorted(teams.dropna().astype(str).unique().tolist())

    def _train_from_test_data(self, df):
        required = FEATURES + ["Bat First", "Bat Second", "Target Score"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError("testData.csv is missing columns: " + ", ".join(missing))

        X_raw = self._prepare_features(df)

        # The original notebook uses Chased Successfully as the class target.
        # Reconstruct it from the winner column when available; otherwise use
        # the batting-second success field.
        if "Chased Successfully" in df.columns:
            y_class = pd.to_numeric(
                df["Chased Successfully"], errors="coerce"
            ).fillna(0).astype(int)
        elif "Winner" in df.columns:
            y_class = (df["Winner"].astype(str) == df["Bat Second"].astype(str)).astype(int)
        else:
            raise ValueError(
                "testData.csv needs either 'Chased Successfully' or 'Winner'."
            )

        y_score = pd.to_numeric(df["Target Score"], errors="coerce").fillna(0).astype(float)

        self.feature_scaler.fit(X_raw)
        X = self.feature_scaler.transform(X_raw)

        self.score_scaler.fit(y_score.to_numpy().reshape(-1, 1))
        y_score_scaled = self.score_scaler.transform(
            y_score.to_numpy().reshape(-1, 1)
        ).ravel()

        self.classifier = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        self.regressor = RandomForestRegressor(
            n_estimators=100, random_state=42
        )

        self.classifier.fit(X, y_class)
        self.regressor.fit(X, y_score_scaled)

        teams = pd.concat([df["Bat First"], df["Bat Second"]], ignore_index=True)
        self.teams = sorted(teams.dropna().astype(str).unique().tolist())

    def predict(
        self,
        bat_first,
        bat_second,
        runs_from_ball,
        innings_runs,
        innings_wickets,
        balls_remaining,
        total_batter_runs,
        total_non_striker_runs,
        batter_balls_faced,
        non_striker_balls_faced,
    ):
        values = [[
            runs_from_ball,
            innings_runs,
            innings_wickets,
            balls_remaining,
            total_batter_runs,
            total_non_striker_runs,
            batter_balls_faced,
            non_striker_balls_faced,
        ]]

        X = self.feature_scaler.transform(values)

        class_prediction = int(self.classifier.predict(X)[0])
        score_scaled = float(self.regressor.predict(X)[0])
        score = float(
            self.score_scaler.inverse_transform([[score_scaled]])[0][0]
        )

        # Same winner interpretation used by the original notebook:
        # class 1 => second batting team, otherwise first batting team.
        winner = bat_second if class_prediction == 1 else bat_first

        probabilities = None
        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(X)[0]
            classes = self.classifier.classes_
            probabilities = {
                int(cls): round(float(p) * 100, 1)
                for cls, p in zip(classes, proba)
            }

        return {
            "winner": winner or "Predicted team",
            "score": round(max(0.0, score), 2),
            "bat_first": bat_first,
            "bat_second": bat_second,
            "probabilities": probabilities,
        }
