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
    Web-deployment version of the Random Forest parts of the
    original MatchResult notebook.

    Uses the same 8 match-state features and MinMaxScaler.
    """

    def __init__(self):
        self.teams = []
        self.training_source = ""

        self.feature_scaler = MinMaxScaler((0, 1))
        self.score_scaler = MinMaxScaler((0, 1))

        dataset_path = BASE_DIR / "Dataset" / "ball_by_ball_it20.csv"
        test_path = BASE_DIR / "testData.csv"
        root_test_path = BASE_DIR / "Dataset" / "testData.csv"

        # ---------------------------------------------------------
        # Load training data
        # ---------------------------------------------------------
        if dataset_path.exists():

            df = pd.read_csv(dataset_path, nrows=30000)

            self.training_source = (
                "Dataset/ball_by_ball_it20.csv (up to 30,000 rows)"
            )

            self._train_from_full_dataset(df)

        elif test_path.exists():

            df = pd.read_csv(test_path)

            self.training_source = "testData.csv fallback (demo mode)"

            self._train_from_test_data(df)

        elif root_test_path.exists():

            df = pd.read_csv(root_test_path)

            self.training_source = (
                "Dataset/testData.csv fallback (demo mode)"
            )

            self._train_from_test_data(df)

        else:

            raise FileNotFoundError(
                "No training data found. Add "
                "Dataset/ball_by_ball_it20.csv or testData.csv "
                "to the project."
            )

    # =============================================================
    # Prepare features
    # =============================================================

    def _prepare_features(self, df):

        missing = [
            column for column in FEATURES
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "Missing feature columns: " + ", ".join(missing)
            )

        X = df[FEATURES].copy().fillna(0)

        return X.astype(float)

    # =============================================================
    # Train using full dataset
    # =============================================================

    def _train_from_full_dataset(self, df):

        required = FEATURES + [
            "Chased Successfully",
            "Target Score",
            "Bat First",
            "Bat Second",
        ]

        missing = [
            column for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "Dataset is missing columns: "
                + ", ".join(missing)
            )

        # Prepare input features
        X_raw = self._prepare_features(df)

        # Classification target
        y_class = (
            pd.to_numeric(
                df["Chased Successfully"],
                errors="coerce"
            )
            .fillna(0)
            .astype(int)
        )

        # Score target
        y_score = (
            pd.to_numeric(
                df["Target Score"],
                errors="coerce"
            )
            .fillna(0)
            .astype(float)
        )

        # ---------------------------------------------------------
        # Feature scaling
        # ---------------------------------------------------------

        self.feature_scaler.fit(X_raw)

        X = self.feature_scaler.transform(X_raw)

        # ---------------------------------------------------------
        # Score scaling
        # ---------------------------------------------------------

        self.score_scaler.fit(
            y_score.to_numpy().reshape(-1, 1)
        )

        y_score_scaled = self.score_scaler.transform(
            y_score.to_numpy().reshape(-1, 1)
        ).ravel()

        # ---------------------------------------------------------
        # Random Forest models
        # ---------------------------------------------------------

        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        self.regressor = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        self.classifier.fit(X, y_class)

        self.regressor.fit(
            X,
            y_score_scaled
        )

        # ---------------------------------------------------------
        # Get team names
        # ---------------------------------------------------------

        teams = pd.concat(
            [
                df["Bat First"],
                df["Bat Second"]
            ],
            ignore_index=True
        )

        self.teams = sorted(
            teams
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    # =============================================================
    # Train using testData.csv fallback
    # =============================================================

    def _train_from_test_data(self, df):

        required = FEATURES + [
            "Bat First",
            "Bat Second",
            "Target Score",
        ]

        missing = [
            column for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "testData.csv is missing columns: "
                + ", ".join(missing)
            )

        X_raw = self._prepare_features(df)

        # ---------------------------------------------------------
        # Classification target
        # ---------------------------------------------------------

        if "Chased Successfully" in df.columns:

            y_class = (
                pd.to_numeric(
                    df["Chased Successfully"],
                    errors="coerce"
                )
                .fillna(0)
                .astype(int)
            )

        elif "Winner" in df.columns:

            y_class = (
                df["Winner"].astype(str)
                ==
                df["Bat Second"].astype(str)
            ).astype(int)

        else:

            raise ValueError(
                "testData.csv needs either "
                "'Chased Successfully' or 'Winner'."
            )

        # ---------------------------------------------------------
        # Score target
        # ---------------------------------------------------------

        y_score = (
            pd.to_numeric(
                df["Target Score"],
                errors="coerce"
            )
            .fillna(0)
            .astype(float)
        )

        # ---------------------------------------------------------
        # Feature scaling
        # ---------------------------------------------------------

        self.feature_scaler.fit(X_raw)

        X = self.feature_scaler.transform(X_raw)

        # ---------------------------------------------------------
        # Score scaling
        # ---------------------------------------------------------

        self.score_scaler.fit(
            y_score.to_numpy().reshape(-1, 1)
        )

        y_score_scaled = self.score_scaler.transform(
            y_score.to_numpy().reshape(-1, 1)
        ).ravel()

        # ---------------------------------------------------------
        # Random Forest models
        # ---------------------------------------------------------

        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        self.regressor = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        self.classifier.fit(
            X,
            y_class
        )

        self.regressor.fit(
            X,
            y_score_scaled
        )

        # ---------------------------------------------------------
        # Get team names
        # ---------------------------------------------------------

        teams = pd.concat(
            [
                df["Bat First"],
                df["Bat Second"]
            ],
            ignore_index=True
        )

        self.teams = sorted(
            teams
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    # =============================================================
    # Prediction
    # =============================================================

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

        # ---------------------------------------------------------
        # Create input row
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # Scale input
        # ---------------------------------------------------------

        X = self.feature_scaler.transform(values)

        # ---------------------------------------------------------
        # Predict winner
        # ---------------------------------------------------------

        class_prediction = int(
            self.classifier.predict(X)[0]
        )

        # Class 0 = Bat First
        # Class 1 = Bat Second

        winner = (
            bat_second
            if class_prediction == 1
            else bat_first
        )

        # ---------------------------------------------------------
        # Predict score
        # ---------------------------------------------------------

        score_scaled = float(
            self.regressor.predict(X)[0]
        )

        score = float(
            self.score_scaler.inverse_transform(
                [[score_scaled]]
            )[0][0]
        )

        # ---------------------------------------------------------
        # Calculate probabilities
        # ---------------------------------------------------------

        probabilities = None

        if hasattr(
            self.classifier,
            "predict_proba"
        ):

            proba = self.classifier.predict_proba(X)[0]

            classes = self.classifier.classes_

            # IMPORTANT:
            # Use team names as dictionary keys instead of
            # 0 and 1. This fixes the duplicate team-name bug
            # on the web page.

            probabilities = {
                bat_first: 0.0,
                bat_second: 0.0
            }

            for cls, probability in zip(
                classes,
                proba
            ):

                if int(cls) == 1:

                    probabilities[bat_second] = round(
                        float(probability) * 100,
                        1
                    )

                else:

                    probabilities[bat_first] = round(
                        float(probability) * 100,
                        1
                    )

        # ---------------------------------------------------------
        # Return result
        # ---------------------------------------------------------

        return {
            "winner": winner or "Predicted team",

            "score": round(
                max(0.0, score),
                2
            ),

            "bat_first": bat_first,

            "bat_second": bat_second,

            "probabilities": probabilities,
        }