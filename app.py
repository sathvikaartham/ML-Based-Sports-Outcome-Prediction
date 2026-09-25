
from flask import Flask, render_template, request
from prediction import Predictor
import math
import os

# =============================================================
# INITIALIZE FLASK APPLICATION
# =============================================================

app = Flask(__name__)

# =============================================================
# INTERNATIONAL TEAMS
# Keep these as plain names, without emoji flags.
# Your prediction model expects the actual team names.
# =============================================================

TEAMS = [
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

# =============================================================
# INITIALIZE PREDICTOR ONCE
# =============================================================

predictor = Predictor()

print("Model:", predictor.best_model_name)
print("Test Accuracy:", predictor.best_model_accuracy)

# =============================================================
# HOME ROUTE
# =============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    selected_bat_first = ""
    selected_bat_second = ""

    if request.method == "POST":

        try:
            # -------------------------------------------------
            # GET TEAM SELECTIONS
            # -------------------------------------------------

            bat_first = request.form.get(
                "bat_first", ""
            ).strip()

            bat_second = request.form.get(
                "bat_second", ""
            ).strip()

            selected_bat_first = bat_first
            selected_bat_second = bat_second

            # -------------------------------------------------
            # VALIDATE TEAM SELECTIONS
            # -------------------------------------------------

            if not bat_first or not bat_second:
                raise ValueError(
                    "Please select both batting and bowling teams."
                )

            if bat_first == bat_second:
                raise ValueError(
                    "Please select two different teams."
                )

            if bat_first not in TEAMS or bat_second not in TEAMS:
                raise ValueError(
                    "Please select teams from the dropdown."
                )

            if (
                bat_first not in predictor.teams
                or bat_second not in predictor.teams
            ):
                raise ValueError(
                    "One of the selected teams is not supported "
                    "by the trained model."
                )

            # -------------------------------------------------
            # GET AND VALIDATE NUMERIC INPUTS
            # -------------------------------------------------

            fields = [
                "runs_from_ball",
                "innings_runs",
                "innings_wickets",
                "balls_remaining",
                "total_batter_runs",
                "total_non_striker_runs",
                "batter_balls_faced",
                "non_striker_balls_faced"
            ]

            values = {}

            for field in fields:
                raw_value = request.form.get(field, "0").strip()

                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"{field.replace('_', ' ').title()} "
                        "must be a valid number."
                    )

                if not math.isfinite(value) or value < 0:
                    raise ValueError(
                        f"{field.replace('_', ' ').title()} "
                        "must be a non-negative number."
                    )

                values[field] = value

            # -------------------------------------------------
            # VALIDATE MATCH STATISTICS
            # -------------------------------------------------

            if values["innings_wickets"] > 10:
                raise ValueError(
                    "Innings wickets cannot exceed 10."
                )

            if not values["innings_wickets"].is_integer():
                raise ValueError(
                    "Innings wickets must be a whole number."
                )

            if values["balls_remaining"] > 120:
                raise ValueError(
                    "Balls remaining cannot exceed 120."
                )

            if not values["balls_remaining"].is_integer():
                raise ValueError(
                    "Balls remaining must be a whole number."
                )

            for field in [
                "batter_balls_faced",
                "non_striker_balls_faced"
            ]:
                if not values[field].is_integer():
                    raise ValueError(
                        f"{field.replace('_', ' ').title()} "
                        "must be a whole number."
                    )

            # -------------------------------------------------
            # MAKE PREDICTION
            # -------------------------------------------------

            result = predictor.predict(
                bat_first=bat_first,
                bat_second=bat_second,
                runs_from_ball=values["runs_from_ball"],
                innings_runs=values["innings_runs"],
                innings_wickets=values["innings_wickets"],
                balls_remaining=values["balls_remaining"],
                total_batter_runs=values["total_batter_runs"],
                total_non_striker_runs=values[
                    "total_non_striker_runs"
                ],
                batter_balls_faced=values[
                    "batter_balls_faced"
                ],
                non_striker_balls_faced=values[
                    "non_striker_balls_faced"
                ]
            )

        except ValueError as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Prediction failed")
            error = (
                "An unexpected error occurred while making "
                "the prediction. Please check your inputs "
                "and try again."
            )

    # =========================================================
    # MODEL INFORMATION
    # =========================================================

    return render_template(
        "index.html",
        teams=TEAMS,
        selected_bat_first=selected_bat_first,
        selected_bat_second=selected_bat_second,
        result=result,
        error=error,
        best_model=predictor.best_model_name,
        best_model_accuracy=predictor.best_model_accuracy,
        model_accuracy=predictor.model_accuracy,
        accuracies=predictor.accuracies,
        training_source=predictor.training_source
    )


# =============================================================
# RUN APPLICATION
# =============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    print("\nStarting Flask server...")
    print(f"Open http://127.0.0.1:{port} in your browser\n")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )