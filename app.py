from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# ---------------- TEAM LIST (30 UNIQUE NBA TEAMS) ---------------- #
TEAM_IDS = [
    1610612737, 1610612738, 1610612739, 1610612740, 1610612741,
    1610612742, 1610612743, 1610612744, 1610612745, 1610612746,
    1610612747, 1610612748, 1610612749, 1610612750, 1610612751,
    1610612752, 1610612753, 1610612754, 1610612755, 1610612756,
    1610612757, 1610612758, 1610612759, 1610612760, 1610612761,
    1610612762, 1610612763, 1610612764, 1610612765, 1610612766
]

# ---------------- LOAD DATA ---------------- #
games = pd.read_csv("games.csv")
games['GAME_DATE_EST'] = pd.to_datetime(games['GAME_DATE_EST'])
games = games.sort_values('GAME_DATE_EST').reset_index(drop=True)

games['HOME_TEAM'] = games['HOME_TEAM_ID']
games['AWAY_TEAM'] = games['VISITOR_TEAM_ID']
games['RESULT'] = games['HOME_TEAM_WINS']

# ---------------- ELO SETTINGS ---------------- #
ELO_START = 1500
K = 20
HOME_ADV = 65

elo = {}

def expected(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))

home_elo, away_elo = [], []

for _, row in games.iterrows():
    h, a = row['HOME_TEAM'], row['AWAY_TEAM']

    elo.setdefault(h, ELO_START)
    elo.setdefault(a, ELO_START)

    Rh = elo[h] + HOME_ADV
    Ra = elo[a]

    home_elo.append(Rh)
    away_elo.append(Ra)

    Eh = expected(Rh, Ra)
    result = row['RESULT']

    elo[h] += K * (result - Eh)
    elo[a] += K * ((1 - result) - (1 - Eh))

games['elo_diff'] = np.array(home_elo) - np.array(away_elo)

# ---------------- MODEL TRAINING ---------------- #
X = games[['elo_diff']]
y = games['RESULT']

split = int(len(games) * 0.8)
X_train, y_train = X.iloc[:split], y.iloc[:split]

model = LogisticRegression()
model.fit(X_train, y_train)

# ---------------- ROUTE ---------------- #
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        home_id = int(request.form["home"])
        away_id = int(request.form["away"])

        if home_id == away_id:
            error = "❌ Home and Visitor teams must be different."
        else:
            elo_diff = elo[home_id] - elo[away_id]
            prob = model.predict_proba([[elo_diff]])[0][1]

            winner = "HOME TEAM WINS 🏠" if prob >= 0.5 else "AWAY TEAM WINS ✈️"

            prediction = {
                "home": home_id,
                "away": away_id,
                "prob": f"{prob*100:.2f}%",
                "winner": winner
            }

    return render_template(
        "index.html",
        teams=TEAM_IDS,
        prediction=prediction,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
