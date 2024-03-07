import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor
from PIL import Image

# Load your model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Define team and city lists
teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka']
cities = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban', 'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion', 'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton', 'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff', 'Christchurch', 'Trinidad']

# Set the background color and page width
st.set_page_config(
    page_title="ICC MEN T20 WORLD CUP SCORE PREDICTOR",
    page_icon="🏏",
    layout="wide",  # You can choose "wide" or "centered"
    initial_sidebar_state="expanded",
)

# Customize the interface with CSS
st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f4;  
        }
        .stApp {
            max-width: 1200px;  
            margin: 0 auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the images
image2 = Image.open("World-cup-T20-2024.png")

# Display the title and image
st.markdown("<h1 style='text-align: center; color: #1f78b4;'>ICC MEN T20 WORLD CUP SCORE PREDICTOR</h1>", unsafe_allow_html=True)
st.image(image2)

# Create columns for user input
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams),index=None)
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams),index=None)

city = st.selectbox('Select city', sorted(cities),index=None)

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score')
with col4:
    overs = st.number_input('Overs done (works for over > 5)')
with col5:
    wickets = st.number_input('Wickets fallen')

last_five = st.number_input('Runs scored in last 5 overs')

if st.button('Predict Score'):
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = current_score / overs

    input_df = pd.DataFrame(
        {'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': city, 'current_score': [current_score],
         'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr], 'last_five': [last_five]})
    result = pipe.predict(input_df)
    st.header("Predicted Score - " + str(int(result[0])))
