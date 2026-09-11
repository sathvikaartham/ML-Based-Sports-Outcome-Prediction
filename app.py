from flask import Flask, render_template, request
from prediction import Predictor

app = Flask(__name__)
predictor = Predictor()

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        try:
            result = predictor.predict(
                bat_first=request.form.get("bat_first", ""),
                bat_second=request.form.get("bat_second", ""),
                runs_from_ball=float(request.form.get("runs_from_ball", 0)),
                innings_runs=float(request.form.get("innings_runs", 0)),
                innings_wickets=float(request.form.get("innings_wickets", 0)),
                balls_remaining=float(request.form.get("balls_remaining", 0)),
                total_batter_runs=float(request.form.get("total_batter_runs", 0)),
                total_non_striker_runs=float(request.form.get("total_non_striker_runs", 0)),
                batter_balls_faced=float(request.form.get("batter_balls_faced", 0)),
                non_striker_balls_faced=float(request.form.get("non_striker_balls_faced", 0)),
            )
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        result=result,
        error=error,
        teams=predictor.teams,
        training_source=predictor.training_source,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
