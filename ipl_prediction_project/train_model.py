import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

matches = pd.read_csv("matches.csv")

# select important columns
data = matches[['team1','team2','venue','winner']].dropna()

# encode categorical data
le_team = LabelEncoder()
le_venue = LabelEncoder()

data['team1'] = le_team.fit_transform(data['team1'])
data['team2'] = le_team.transform(data['team2'])
data['venue'] = le_venue.fit_transform(data['venue'])
data['winner'] = le_team.transform(data['winner'])

X = data[['team1','team2','venue']]
y = data['winner']

model = LogisticRegression(max_iter=1000)
model.fit(X,y)

pickle.dump(model, open("model.pkl","wb"))
pickle.dump(le_team, open("team_encoder.pkl","wb"))
pickle.dump(le_venue, open("venue_encoder.pkl","wb"))

print("Model trained successfully")