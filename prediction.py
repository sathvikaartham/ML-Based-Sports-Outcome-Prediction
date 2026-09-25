
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


class Predictor:

    def __init__(self):

        # Main international teams
        self.teams = [
            "Afghanistan",
            "Australia",
            "Bangladesh",
            "England",
            "India",
            "Ireland",
            "Netherlands",
            "New Zealand",
            "Pakistan",
            "South Africa",
            "Sri Lanka",
            "Zimbabwe"
        ]

        self.best_model_name = "Random Forest"
        self.best_model_accuracy = 0
        self.model_accuracy = 0
        self.accuracies = {}
        self.training_source = "Dataset/ball_by_ball_it20.csv"

        self.model = None
        self.team_encoder = LabelEncoder()

        self._load_and_train()

    def _load_and_train(self):

        # Dataset path
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            "Dataset",
            "ball_by_ball_it20.csv"
        )

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}"
            )

        df = pd.read_csv(dataset_path)

        print("Dataset loaded successfully.")
        print("Dataset shape:", df.shape)

        # Normalize column names
        df.columns = [
            str(col).strip().lower().replace(" ", "_")
            for col in df.columns
        ]

        # Check required columns
        required_columns = [
            "bat_first",
            "bat_second",
            "winner"
        ]

        for column in required_columns:
            if column not in df.columns:
                raise ValueError(
                    f"Missing required column: {column}. "
                    f"Available columns: {list(df.columns)}"
                )

        # Keep only matches involving selected teams
        df = df[
            df["bat_first"].isin(self.teams)
            & df["bat_second"].isin(self.teams)
            & df["winner"].isin(self.teams)
        ].copy()

        if df.empty:
            raise ValueError(
                "No matching international team records "
                "were found in the dataset."
            )

        # Convert team names into numerical values
        self.team_encoder.fit(self.teams)

        df["bat_first_encoded"] = (
            self.team_encoder.transform(df["bat_first"])
        )

        df["bat_second_encoded"] = (
            self.team_encoder.transform(df["bat_second"])
        )

        df["winner_encoded"] = (
            self.team_encoder.transform(df["winner"])
        )

        # Features and target
        feature_columns = [
            "bat_first_encoded",
            "bat_second_encoded"
        ]

        # Include available numerical match statistics
        possible_features = [
            "runs_from_ball",
            "innings_runs",
            "innings_wickets",
            "balls_remaining",
            "total_batter_runs",
            "total_non_striker_runs",
            "batter_balls_faced",
            "non_striker_balls_faced"
        ]

        for column in possible_features:
            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                ).fillna(0)

                feature_columns.append(column)

        X = df[feature_columns]
        y = df["winner_encoded"]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        # Evaluate model
        predictions = self.model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        ) * 100

        self.model_accuracy = round(accuracy, 2)
        self.best_model_accuracy = self.model_accuracy

        self.accuracies = {
            "Random Forest": self.model_accuracy
        }

        self.feature_columns = feature_columns

        print("Model training completed.")
        print("Model:", self.best_model_name)
        print("Accuracy:", self.model_accuracy, "%")

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

        # Validate teams
        if bat_first not in self.teams:
            raise ValueError("Invalid batting team.")

        if bat_second not in self.teams:
            raise ValueError("Invalid bowling team.")

        if bat_first == bat_second:
            raise ValueError(
                "Batting and bowling teams must be different."
            )

        # Encode teams
        first_encoded = self.team_encoder.transform(
            [bat_first]
        )[0]

        second_encoded = self.team_encoder.transform(
            [bat_second]
        )[0]

        # Prepare prediction input
        input_data = {
            "bat_first_encoded": first_encoded,
            "bat_second_encoded": second_encoded,
            "runs_from_ball": float(runs_from_ball),
            "innings_runs": float(innings_runs),
            "innings_wickets": float(innings_wickets),
            "balls_remaining": float(balls_remaining),
            "total_batter_runs": float(total_batter_runs),
            "total_non_striker_runs": float(
                total_non_striker_runs
            ),
            "batter_balls_faced": float(
                batter_balls_faced
            ),
            "non_striker_balls_faced": float(
                non_striker_balls_faced
            )
        }

        input_df = pd.DataFrame(
            [input_data],
            columns=self.feature_columns
        )

        # Predict winner
        predicted_class = self.model.predict(
            input_df
        )[0]

        winner = self.team_encoder.inverse_transform(
            [predicted_class]
        )[0]

        # Get prediction probabilities
        raw_probabilities = self.model.predict_proba(
            input_df
        )[0]

        probabilities = {}

        for index, class_value in enumerate(
            self.model.classes_
        ):
            team_name = self.team_encoder.inverse_transform(
                [int(class_value)]
            )[0]

            probabilities[team_name] = round(
                float(raw_probabilities[index]) * 100,
                2
            )

        # Current score
        score = int(innings_runs)

        return {
            "winner": winner,
            "score": score,
            "probabilities": probabilities
        }