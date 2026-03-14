from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
team_encoder = pickle.load(open("team_encoder.pkl","rb"))
venue_encoder = pickle.load(open("venue_encoder.pkl","rb"))

@app.route("/", methods=["GET","POST"])
def home():

    winner=None

    if request.method=="POST":

        batting_team = request.form["batting_team"]
        bowling_team = request.form["bowling_team"]
        venue = request.form["venue"]

        team1 = team_encoder.transform([batting_team])[0]
        team2 = team_encoder.transform([bowling_team])[0]
        venue = venue_encoder.transform([venue])[0]

        prediction = model.predict([[team1,team2,venue]])

        winner = team_encoder.inverse_transform(prediction)[0]

    return render_template("index.html",winner=winner)


if __name__ == "__main__":
    app.run(debug=True)