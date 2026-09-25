
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score


# =============================================================
# PROJECT CONFIGURATION
# =============================================================

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


# =============================================================
# PREDICTOR CLASS
# =============================================================

class Predictor:

    def __init__(self):

        # -----------------------------------------------------
        # INITIALIZE ALL ATTRIBUTES
        # -----------------------------------------------------

        self.teams = []

        self.training_source = ""

        self.accuracies = {}

        self.best_model_name = "Random Forest Classifier"

        self.best_model_accuracy = None

        self.model_accuracy = None

        self.classifier = None

        self.regressor = None

        self.feature_scaler = MinMaxScaler(
            feature_range=(0, 1)
        )

        self.score_scaler = MinMaxScaler(
            feature_range=(0, 1)
        )

        # -----------------------------------------------------
        # DATASET PATHS
        # -----------------------------------------------------

        dataset_path = (
            BASE_DIR
            / "Dataset"
            / "ball_by_ball_it20.csv"
        )

        test_path = (
            BASE_DIR / "testData.csv"
        )

        root_test_path = (
            BASE_DIR
            / "Dataset"
            / "testData.csv"
        )

        # -----------------------------------------------------
        # LOAD DATASET
        # -----------------------------------------------------

        if dataset_path.exists():

            df = pd.read_csv(
                dataset_path
            )

            self.training_source = (
                "Dataset/ball_by_ball_it20.csv"
            )

            self._train_from_full_dataset(df)

        elif test_path.exists():

            df = pd.read_csv(
                test_path
            )

            self.training_source = (
                "testData.csv fallback (demo mode)"
            )

            self._train_from_test_data(df)

        elif root_test_path.exists():

            df = pd.read_csv(
                root_test_path
            )

            self.training_source = (
                "Dataset/testData.csv fallback "
                "(demo mode)"
            )

            self._train_from_test_data(df)

        else:

            raise FileNotFoundError(
                "No training dataset found. Please add "
                "Dataset/ball_by_ball_it20.csv "
                "or testData.csv to your project."
            )

    # =========================================================
    # PREPARE FEATURES
    # =========================================================

    def _prepare_features(self, df):

        missing = [
            column
            for column in FEATURES
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                "Missing feature columns: "
                + ", ".join(missing)
            )

        X = df[FEATURES].copy()

        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        X = X.fillna(0)

        return X.astype(float)

    # =========================================================
    # PREPARE CLASSIFICATION TARGET
    # =========================================================

    def _prepare_classification_target(self, df):

        if "Chased Successfully" in df.columns:

            y = pd.to_numeric(
                df["Chased Successfully"],
                errors="coerce"
            )

        elif "Winner" in df.columns:

            winner = (
                df["Winner"]
                .astype(str)
                .str.strip()
            )

            second_team = (
                df["Bat Second"]
                .astype(str)
                .str.strip()
            )

            y = (
                winner == second_team
            ).astype(int)

        else:

            raise ValueError(
                "Dataset requires either "
                "'Chased Successfully' or 'Winner'."
            )

        return y

    # =========================================================
    # PREPARE SCORE TARGET
    # =========================================================

    def _prepare_score_target(self, df):

        return pd.to_numeric(
            df["Target Score"],
            errors="coerce"
        )

    # =========================================================
    # GET TEAM NAMES
    # =========================================================

    def _get_teams(self, df):

        teams = pd.concat(
            [
                df["Bat First"],
                df["Bat Second"]
            ],
            ignore_index=True
        )

        teams = (
            teams
            .dropna()
            .astype(str)
            .str.strip()
        )

        teams = teams[
            teams != ""
        ]

        self.teams = sorted(
            teams.unique().tolist()
        )

    # =========================================================
    # EVALUATE MODEL ACCURACY
    # =========================================================

    def _evaluate_classifier(self, X, y):

        # -----------------------------------------------------
        # INITIALIZE ACCURACY ATTRIBUTES
        # -----------------------------------------------------

        self.best_model_accuracy = None

        self.model_accuracy = None

        self.accuracies = {
            self.best_model_name: None
        }

        # -----------------------------------------------------
        # CHECK DATA AVAILABILITY
        # -----------------------------------------------------

        class_counts = y.value_counts()

        if (
            len(X) < 10
            or len(class_counts) < 2
            or class_counts.min() < 2
        ):

            print(
                "Warning: Insufficient data to "
                "calculate reliable test accuracy."
            )

            return

        # -----------------------------------------------------
        # SPLIT DATA
        # -----------------------------------------------------

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )
        )

        # -----------------------------------------------------
        # SCALE TRAINING DATA
        # -----------------------------------------------------

        evaluation_scaler = MinMaxScaler(
            feature_range=(0, 1)
        )

        X_train_scaled = (
            evaluation_scaler.fit_transform(
                X_train
            )
        )

        X_test_scaled = (
            evaluation_scaler.transform(
                X_test
            )
        )

        # -----------------------------------------------------
        # TRAIN EVALUATION MODEL
        # -----------------------------------------------------

        evaluation_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )

        evaluation_model.fit(
            X_train_scaled,
            y_train
        )

        # -----------------------------------------------------
        # PREDICT TEST DATA
        # -----------------------------------------------------

        y_pred = (
            evaluation_model.predict(
                X_test_scaled
            )
        )

        # -----------------------------------------------------
        # CALCULATE ACCURACY
        # -----------------------------------------------------

        accuracy = (
            accuracy_score(
                y_test,
                y_pred
            ) * 100
        )

        self.best_model_accuracy = round(
            float(accuracy),
            2
        )

        self.model_accuracy = (
            self.best_model_accuracy
        )

        self.accuracies = {
            self.best_model_name:
                self.best_model_accuracy
        }

        print(
            f"Model: {self.best_model_name}"
        )

        print(
            f"Test Accuracy: "
            f"{self.best_model_accuracy}%"
        )

    # =========================================================
    # TRAIN FINAL MODELS
    # =========================================================

    def _train_models(self, df):

        # -----------------------------------------------------
        # PREPARE FEATURES AND TARGETS
        # -----------------------------------------------------

        X_raw = self._prepare_features(
            df
        )

        y_class = (
            self._prepare_classification_target(
                df
            )
        )

        y_score = (
            self._prepare_score_target(
                df
            )
        )

        # -----------------------------------------------------
        # REMOVE INVALID TARGET ROWS
        # -----------------------------------------------------

        valid_rows = (
            y_class.notna()
            & y_score.notna()
            & np.isfinite(y_class)
            & np.isfinite(y_score)
        )

        X_raw = (
            X_raw.loc[valid_rows]
            .reset_index(drop=True)
        )

        y_class = (
            y_class.loc[valid_rows]
            .reset_index(drop=True)
            .astype(int)
        )

        y_score = (
            y_score.loc[valid_rows]
            .reset_index(drop=True)
            .astype(float)
        )

        if len(X_raw) < 2:

            raise ValueError(
                "Not enough valid data to train "
                "the prediction model."
            )

        if not y_class.isin([0, 1]).all():

            raise ValueError(
                "Classification target must contain "
                "only 0 and 1 values."
            )

        # -----------------------------------------------------
        # CALCULATE ACCURACY
        # -----------------------------------------------------

        self._evaluate_classifier(
            X_raw,
            y_class
        )

        # -----------------------------------------------------
        # TRAIN FINAL FEATURE SCALER
        # -----------------------------------------------------

        self.feature_scaler.fit(
            X_raw
        )

        X_scaled = (
            self.feature_scaler.transform(
                X_raw
            )
        )

        # -----------------------------------------------------
        # TRAIN SCORE SCALER
        # -----------------------------------------------------

        self.score_scaler.fit(
            y_score.to_numpy().reshape(-1, 1)
        )

        y_score_scaled = (
            self.score_scaler.transform(
                y_score.to_numpy().reshape(-1, 1)
            ).ravel()
        )

        # -----------------------------------------------------
        # TRAIN FINAL RANDOM FOREST CLASSIFIER
        # -----------------------------------------------------

        self.classifier = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )

        self.classifier.fit(
            X_scaled,
            y_class
        )

        # -----------------------------------------------------
        # TRAIN FINAL RANDOM FOREST REGRESSOR
        # -----------------------------------------------------

        self.regressor = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        self.regressor.fit(
            X_scaled,
            y_score_scaled
        )

        # -----------------------------------------------------
        # GET TEAM NAMES
        # -----------------------------------------------------

        self._get_teams(
            df
        )

        print(
            "Training completed successfully."
        )

        print(
            f"Training rows: {len(X_raw)}"
        )

        print(
            f"Selected model: "
            f"{self.best_model_name}"
        )

        print(
            f"Model accuracy: "
            f"{self.best_model_accuracy}"
        )

    # =========================================================
    # TRAIN USING FULL DATASET
    # =========================================================

    def _train_from_full_dataset(self, df):

        required = FEATURES + [
            "Chased Successfully",
            "Target Score",
            "Bat First",
            "Bat Second"
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                "Full dataset is missing columns: "
                + ", ".join(missing)
            )

        self._train_models(
            df
        )

    # =========================================================
    # TRAIN USING TEST DATA FALLBACK
    # =========================================================

    def _train_from_test_data(self, df):

        required = FEATURES + [
            "Bat First",
            "Bat Second",
            "Target Score"
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                "testData.csv is missing columns: "
                + ", ".join(missing)
            )

        self._train_models(
            df
        )

    # =========================================================
    # PREDICTION
    # =========================================================

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
        non_striker_balls_faced
    ):

        # -----------------------------------------------------
        # VALIDATE TEAMS
        # -----------------------------------------------------

        bat_first = str(
            bat_first
        ).strip()

        bat_second = str(
            bat_second
        ).strip()

        if not bat_first or not bat_second:

            raise ValueError(
                "Please select both teams."
            )

        if bat_first == bat_second:

            raise ValueError(
                "Batting Team and Bowling Team "
                "must be different."
            )

        if bat_first not in self.teams:

            raise ValueError(
                "Invalid batting team selected."
            )

        if bat_second not in self.teams:

            raise ValueError(
                "Invalid bowling team selected."
            )

        # -----------------------------------------------------
        # CREATE INPUT DATA
        # -----------------------------------------------------

        values = pd.DataFrame(
            [[
                runs_from_ball,
                innings_runs,
                innings_wickets,
                balls_remaining,
                total_batter_runs,
                total_non_striker_runs,
                batter_balls_faced,
                non_striker_balls_faced
            ]],
            columns=FEATURES
        )

        values = values.apply(
            pd.to_numeric,
            errors="coerce"
        )

        values = values.replace(
            [np.inf, -np.inf],
            np.nan
        )

        if values.isna().any().any():

            raise ValueError(
                "Please enter valid numeric values."
            )

        if (values < 0).any().any():

            raise ValueError(
                "Match statistics cannot be negative."
            )

        # -----------------------------------------------------
        # SCALE INPUT
        # -----------------------------------------------------

        X = (
            self.feature_scaler.transform(
                values
            )
        )

        # -----------------------------------------------------
        # PREDICT WINNER
        # -----------------------------------------------------

        class_prediction = int(
            self.classifier.predict(X)[0]
        )

        # Class 0 = Bat First
        # Class 1 = Bat Second

        if class_prediction == 1:

            winner = bat_second

        else:

            winner = bat_first

        # -----------------------------------------------------
        # PREDICT SCORE
        # -----------------------------------------------------

        score_scaled = float(
            self.regressor.predict(X)[0]
        )

        score = float(
            self.score_scaler.inverse_transform(
                [[score_scaled]]
            )[0][0]
        )

        score = round(
            max(0.0, score),
            2
        )

        # -----------------------------------------------------
        # CALCULATE PROBABILITIES
        # -----------------------------------------------------

        probabilities = {
            bat_first: 0.0,
            bat_second: 0.0
        }

        if hasattr(
            self.classifier,
            "predict_proba"
        ):

            proba = (
                self.classifier.predict_proba(X)[0]
            )

            classes = (
                self.classifier.classes_
            )

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

        # -----------------------------------------------------
        # RETURN RESULTS
        # -----------------------------------------------------

        return {

            "winner": winner,

            "score": score,

            "bat_first": bat_first,

            "bat_second": bat_second,

            "probabilities": probabilities,

            "best_model": self.best_model_name,

            "best_model_accuracy":
                self.best_model_accuracy,

            "model_accuracy":
                self.model_accuracy
        }