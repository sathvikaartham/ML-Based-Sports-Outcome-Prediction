# ML-Based Sports Outcome Prediction — Web App

This folder adds a deployable Flask web interface to the original cricket match
outcome prediction project.

## Files

- `app.py` — Flask web application
- `prediction.py` — Random Forest prediction and preprocessing logic
- `templates/index.html` — frontend
- `static/style.css` — styling
- `requirements.txt` — deployment dependencies
- `Procfile` — Render start command

## Important dataset note

The original notebook trains on:

`Dataset/ball_by_ball_it20.csv` (up to 30,000 rows)

If that file is present, the web app uses it and follows the notebook's
Random Forest preprocessing approach.

If the full dataset is not present, the app falls back to `testData.csv` so
the web demo can still run. This fallback is only a small demo dataset and is
not equivalent to training on the full original dataset.

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 app.py
```

Open:

`http://localhost:5000`

## Render

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
gunicorn app:app
```

## Model inputs

The predictor uses the same eight match-state features from the original
project:

- Runs From Ball
- Innings Runs
- Innings Wickets
- Balls Remaining
- Total Batter Runs
- Total Non Striker Runs
- Batter Balls Faced
- Non Striker Balls Faced

Team names are displayed by the web interface, but they are not numerical
features in the original model; the prediction is based on the match-state
statistics.
