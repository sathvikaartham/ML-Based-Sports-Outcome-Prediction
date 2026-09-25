
from flask import Flask, render_template, request
from prediction import Predictor
import math

# =============================================================
# INITIALIZE FLASK APPLICATION
# =============================================================

app = Flask(__name__)

# =============================================================
# INITIALIZE PREDICTOR
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

    # =========================================================
    # HANDLE POST REQUEST
    # =========================================================

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
                    "Please select both batting "
                    "and bowling teams."
                )

            if bat_first == bat_second:
                raise ValueError(
                    "Batting Team and Bowling Team "
                    "cannot be the same. Please "
                    "select two different teams."
                )

            if (
                bat_first not in predictor.teams
                or bat_second not in predictor.teams
            ):
                raise ValueError(
                    "Invalid team selection. "
                    "Please select teams from "
                    "the dropdown."
                )

            # -------------------------------------------------
            # GET MATCH STATISTICS
            # -------------------------------------------------

            runs_from_ball = float(
                request.form.get("runs_from_ball", 0)
            )

            innings_runs = float(
                request.form.get("innings_runs", 0)
            )

            innings_wickets = float(
                request.form.get("innings_wickets", 0)
            )

            balls_remaining = float(
                request.form.get("balls_remaining", 0)
            )

            total_batter_runs = float(
                request.form.get("total_batter_runs", 0)
            )

            total_non_striker_runs = float(
                request.form.get(
                    "total_non_striker_runs", 0
                )
            )

            batter_balls_faced = float(
                request.form.get(
                    "batter_balls_faced", 0
                )
            )

            non_striker_balls_faced = float(
                request.form.get(
                    "non_striker_balls_faced", 0
                )
            )

            # -------------------------------------------------
            # VALIDATE NUMERIC INPUTS
            # -------------------------------------------------

            numeric_values = {
                "Runs From Ball": runs_from_ball,
                "Innings Runs": innings_runs,
                "Innings Wickets": innings_wickets,
                "Balls Remaining": balls_remaining,
                "Total Batter Runs": total_batter_runs,
                "Total Non-Striker Runs":
                    total_non_striker_runs,
                "Batter Balls Faced":
                    batter_balls_faced,
                "Non-Striker Balls Faced":
                    non_striker_balls_faced
            }

            for field, value in numeric_values.items():

                if not math.isfinite(value) or value < 0:
                    raise ValueError(
                        f"{field} must be a valid "
                        "non-negative number."
                    )

            # -------------------------------------------------
            # VALIDATE WICKETS
            # -------------------------------------------------

            if innings_wickets > 10:
                raise ValueError(
                    "Innings wickets cannot exceed 10."
                )

            if innings_wickets != int(innings_wickets):
                raise ValueError(
                    "Innings wickets must be "
                    "a whole number."
                )

            # -------------------------------------------------
            # VALIDATE BALLS REMAINING
            # -------------------------------------------------

            if balls_remaining > 120:
                raise ValueError(
                    "Balls remaining cannot exceed "
                    "120 for a standard T20 innings."
                )

            if balls_remaining != int(balls_remaining):
                raise ValueError(
                    "Balls remaining must be "
                    "a whole number."
                )

            # -------------------------------------------------
            # VALIDATE BATTER BALLS FACED
            # -------------------------------------------------

            if batter_balls_faced != int(batter_balls_faced):
                raise ValueError(
                    "Batter balls faced must be "
                    "a whole number."
                )

            if (
                non_striker_balls_faced
                != int(non_striker_balls_faced)
            ):
                raise ValueError(
                    "Non-striker balls faced must be "
                    "a whole number."
                )

            # -------------------------------------------------
            # MAKE PREDICTION
            # -------------------------------------------------

            result = predictor.predict(
                bat_first=bat_first,
                bat_second=bat_second,
                runs_from_ball=runs_from_ball,
                innings_runs=innings_runs,
                innings_wickets=innings_wickets,
                balls_remaining=balls_remaining,
                total_batter_runs=total_batter_runs,
                total_non_striker_runs=
                    total_non_striker_runs,
                batter_balls_faced=
                    batter_balls_faced,
                non_striker_balls_faced=
                    non_striker_balls_faced
            )

        # =====================================================
        # ERROR HANDLING
        # =====================================================

        except ValueError as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Prediction failed")

            error = (
                "An unexpected error occurred "
                "while making the prediction. "
                "Please check your inputs and "
                "try again."
            )

    # =========================================================
    # GET MODEL INFORMATION
    # =========================================================

    best_model = predictor.best_model_name

    best_model_accuracy = (
        predictor.best_model_accuracy
    )

    model_accuracy = predictor.model_accuracy

    accuracies = predictor.accuracies

    training_source = predictor.training_source

    # =========================================================
    # RENDER TEMPLATE
    # =========================================================

    return render_template(
        "index.html",

        # TEAM DROPDOWNS
        teams=predictor.teams,
        selected_bat_first=selected_bat_first,
        selected_bat_second=selected_bat_second,

        # PREDICTION RESULTS
        result=result,
        error=error,

        # MODEL INFORMATION
        best_model=best_model,
        best_model_accuracy=best_model_accuracy,
        model_accuracy=model_accuracy,
        accuracies=accuracies,

        # DATASET INFORMATION
        training_source=training_source
    )


# =============================================================
# RUN APPLICATION
# =============================================================

if __name__ == "__main__":

    print("\nStarting Flask server...")
    print("Open http://127.0.0.1:5000 in your browser\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )