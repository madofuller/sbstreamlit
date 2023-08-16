import streamlit as st
import pandas as pd
import scikit as sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import GradientBoostingRegressor
import requests
from bs4 import BeautifulSoup

# Streamlit UI
st.title('Baseball Betting Model and Predictions')

# Load the data
df = pd.read_csv("C:\\NBA_Betting\\ModelTrainingandTestingData.csv")

# Display the first few rows and column names
st.write("Sample data:")
st.dataframe(df.head())
st.write("Column names:")
st.write(list(df.columns))

# Define features and target
features = ['H_WIN%', 'H_PTS_PG', 'H_FGM_PG', ...]  # Your list of features
target = 'Point Total'

# ... (previous code remains the same)

# Load betting data
Bet = pd.read_csv("C:\\NBA_Betting\\328Games.csv")
Bet = Bet.iloc[:, 2:]
Bet = Bet.dropna()

# Web scraping using BeautifulSoup
url = "https://www.baseball-reference.com/leagues/majors/2023-schedule.shtml"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract game data
game_data = []
for row in soup.select('table#schedule tbody tr'):
    cols = row.find_all('td')
    if len(cols) >= 5:
        date = cols[0].get_text()
        home_team = cols[3].get_text()
        away_team = cols[2].get_text()
        game_data.append({'Date': date, 'HomeTeam': home_team, 'AwayTeam': away_team})

# Create a DataFrame for scraped game data
scraped_data = pd.DataFrame(game_data)

# Merge with the betting data
Bet = Bet.merge(scraped_data, how='left', on=['Date', 'HomeTeam', 'AwayTeam'])

# ... (continue with the rest of the script)

# Predict using the trained model
GB_predict_Tonight = GB.predict(Bet)

# Create a DataFrame for tonight's totals
Tonights_Totals = pd.DataFrame(GB_predict_Tonight, columns=['scores'])

# Display the predicted scores
st.write("Predicted scores for tonight:")
st.dataframe(Tonights_Totals)

