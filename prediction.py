
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =============================================================
# PATHS
# =============================================================

BASE_DIR = Path(__file__).resolve().parent

# Load the smaller candidate model.
# The original models.joblib remains untouched.
MODEL_PATH = (
    BASE_DIR
    / "saved_models"
    / "models_small_candidate.joblib"
)


# =============================================================
# FEATURES
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


# =============================================================
# PREDICTOR
# =============================================================

class Predictor:

    def __init__(self):

        # -----------------------------------------------------
        # CHECK MODEL FILE
        # -----------------------------------------------------

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Saved model not found: {MODEL_PATH}. "
                "Run train_model_small.py locally first."
            )

        # -----------------------------------------------------
        # LOAD MODEL ARTIFACTS
        # -----------------------------------------------------

        artifacts = joblib.load(MODEL_PATH)

        self.classifier = artifacts["classifier"]
        self.regressor = artifacts["regressor"]

        self.feature_scaler = artifacts["feature_scaler"]
        self.score_scaler = artifacts["score_scaler"]

        self.teams = artifacts["teams"]

        self.best_model_name = artifacts["model_name"]
        self.best_model_accuracy = artifacts["accuracy"]

        self.model_accuracy = self.best_model_accuracy

        self.accuracies = {
            self.best_model_name: self.best_model_accuracy
        }

        self.training_source = artifacts["training_source"]

        # -----------------------------------------------------
        # STARTUP INFORMATION
        # -----------------------------------------------------

        print("Saved models loaded successfully.")
        print("Model file:", MODEL_PATH.name)
        print("Model:", self.best_model_name)
        print("Teams:", len(self.teams))
        print("Accuracy:", self.best_model_accuracy)


    # =========================================================
    # PREDICT MATCH OUTCOME
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
        # VALIDATE TEAM SELECTIONS
        # -----------------------------------------------------

        bat_first = str(bat_first).strip()
        bat_second = str(bat_second).strip()

        if not bat_first or not bat_second:
            raise ValueError(
                "Please select both teams."
            )

        if bat_first == bat_second:
            raise ValueError(
                "Batting Team and Bowling Team must be different."
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
        # PREPARE INPUT FEATURES
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

        # -----------------------------------------------------
        # VALIDATE NUMERIC INPUTS
        # -----------------------------------------------------

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
        # SCALE FEATURES
        # -----------------------------------------------------

        X = self.feature_scaler.transform(values)

        # -----------------------------------------------------
        # CLASSIFY MATCH OUTCOME
        # -----------------------------------------------------

        class_prediction = int(
            self.classifier.predict(X)[0]
        )

        winner = (
            bat_second if class_prediction == 1
            else bat_first
        )

        # -----------------------------------------------------
        # CALCULATE CLASS PROBABILITIES
        # -----------------------------------------------------

        probabilities = {
            bat_first: 0.0,
            bat_second: 0.0
        }

        proba = self.classifier.predict_proba(X)[0]
        classes = self.classifier.classes_

        for cls, probability in zip(classes, proba):

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
        # RETURN RESULTS
        # -----------------------------------------------------

        return {
            "winner": winner,
            "score": score,
            "bat_first": bat_first,
            "bat_second": bat_second,
            "probabilities": probabilities,
            "best_model": self.best_model_name,
            "best_model_accuracy": self.best_model_accuracy,
            "model_accuracy": self.model_accuracy
        }