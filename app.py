import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import font_manager
from highlight_text import fig_text
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from scipy import stats
import plotly.graph_objs as go
from math import pi
from mplsoccer import Pitch
from mplsoccer import PyPizza
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_gsheets import GSheetsConnection
from mplsoccer.pitch import Pitch, VerticalPitch
import io
import base64
import requests
from io import BytesIO

st.set_page_config(layout="wide")

pd.set_option("display.width", None)  # None means no width limit

# Function to read data from Google Sheets and display it
def display_data():
    
    # Create a connection object.
    url = "https://docs.google.com/spreadsheets/d/1GAghNSTYJTVVl4I9Q-qOv_PGikuj_TQIgSp2sGXz5XM/edit?usp=sharing"

    conn = st.connection("gsheets", type=GSheetsConnection)

    data = conn.read(spreadsheet=url)

    # Convert 'Contract' column to a consistent type (e.g., string)
    data['Contract'] = data['Contract'].astype(str)

    # Extract unique entries in 'Contract' column
    unique_contracts = data['Contract'].unique()

    # Set the default selected expiry date to 2030
    selected_expiry_date = '2030'  # Assuming '2030' is a valid contract expiry date in your data

    # Add a sidebar dropdown box for selecting contract expiry date
    selected_expiry_date = st.sidebar.selectbox("Contract Expiry Before", sorted(unique_contracts), index=unique_contracts.tolist().index(selected_expiry_date))

    # Add a sidebar checkbox to select or exclude players from Stoke City
    include_stoke_city = st.sidebar.checkbox("Include Stoke City players", False)

    # Add a sidebar checkbox to select only players available for loan
    only_loan_available = st.sidebar.checkbox("Select only players available for loan", False)

    # Add a sidebar slider for selecting age range
    min_age, max_age = st.sidebar.slider("Select Age Range", min_value=data['Age'].min(), max_value=data['Age'].max(), value=(data['Age'].min(), data['Age'].max()))

    # Add a sidebar checkbox for selecting only domestic players
    domestic_only = st.sidebar.checkbox("Domestic players only?", False, help="Check to include only domestic players")

    # Filter data for players with contract expiry before selected date
    filtered_data = data[data['Contract'] < selected_expiry_date]

    # Filter data to include or exclude Stoke City players based on user choice
    if not include_stoke_city:
        filtered_data = filtered_data[filtered_data['Current Club'] != 'Stoke City']

    # Filter data by age range
    filtered_data = filtered_data[(filtered_data['Age'] >= min_age) & (filtered_data['Age'] <= max_age)]

    # Filter data based on 'Domestic / Abroad' column
    if domestic_only:
        filtered_data = filtered_data[filtered_data['Domestic / Abroad'] == 'Domestic']

    # Filter data to include only players available for loan if checkbox is checked
    if only_loan_available:
        filtered_data = filtered_data[filtered_data['Available for Loan?'] == 'Yes']

    # Filter data for RB, LB, LW, RW, DM, CM, AM, and ST positions
    rb_data = filtered_data[filtered_data['Position'] == 'RB']
    lb_data = filtered_data[filtered_data['Position'] == 'LB']
    lw_data = filtered_data[filtered_data['Position'] == 'LW']
    rw_data = filtered_data[filtered_data['Position'] == 'RW']
    dm_data = filtered_data[filtered_data['Position'] == 'CDM']
    cm_data = filtered_data[filtered_data['Position'] == 'CM']
    am_data = filtered_data[filtered_data['Position'] == 'AM']
    st_data = filtered_data[filtered_data['Position'] == 'CF']
    cb_data = filtered_data[filtered_data['Position'] == 'CB']  # New: Filter data for CBs

    # Filter CBs for left-footed and right-footed players
    left_footed_cb_data = cb_data[cb_data['Foot'] == 'L']
    right_footed_cb_data = cb_data[cb_data['Foot'] == 'R']

    # Select top 5 players for each position based on some criteria (for example, confidence score)
    top_5_rb_players = rb_data.sort_values(by='Confidence Score', ascending=False).head(5)
    top_5_lb_players = lb_data.sort_values(by='Confidence Score', ascending=False).head(5)
    top_5_lw_players = lw_data.sort_values(by='Confidence Score', ascending=False).head(5)
    top_5_rw_players = rw_data.sort_values(by='Confidence Score', ascending=False).head(5)
    top_5_dm_players = dm_data.sort_values(by='Confidence Score', ascending=False).head(5)
    top_5_cm_players = cm_data.sort_values(by='Confidence Score', ascending=False).head(5)
    top_5_am_players = am_data.sort_values(by='Confidence Score', ascending=False).head(5)
    top_5_st_players = st_data.sort_values(by='Confidence Score', ascending=False).head(5)
    top_5_left_footed_cb_players = left_footed_cb_data.sort_values(by='Confidence Score', ascending=False).head(5)  # New: Top 5 left-footed CBs
    top_5_right_footed_cb_players = right_footed_cb_data.sort_values(by='Confidence Score', ascending=False).head(5)  # New: Top 5 right-footed CBs

    # Plot the top 5 players for each position on the pitch visualization
    plot_players_on_pitch(top_5_rb_players, top_5_lb_players, top_5_lw_players, top_5_rw_players, top_5_dm_players, top_5_cm_players, top_5_am_players, top_5_st_players, top_5_left_footed_cb_players, top_5_right_footed_cb_players, filtered_data, data.columns)

    # Filter and select desired columns
    selected_columns = ["Player", "Current Club", "Position", "Contract", "Confidence Score Last Month"]
    filtered_data = data[selected_columns]

    # Sort the data by 'Confidence Score Last Month'
    sorted_original_data = filtered_data.sort_values(by='Confidence Score Last Month', ascending=False)

    # Center align the text using HTML
    st.markdown("<h3 style='text-align: center;'>Confidence Score Last Month</h3>", unsafe_allow_html=True)

    # Create three columns with adjusted widths
    col1, col2, col3 = st.columns([0.8, 3, 0.8])

    # Display the table in the middle column
    with col2:
        st.dataframe(sorted_original_data, hide_index=True)

# Plotting function
def plot_players_on_pitch(rb_players_data, lb_players_data, lw_players_data, rw_players_data, dm_players_data, cm_players_data, am_players_data, st_players_data, left_cb_players_data, right_cb_players_data, data, column_names):
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#ffffff', stripe=False, line_zorder=2, pad_top=0.1)

    fig, ax = pitch.draw(figsize=(12, 8))  # Adjust the figsize parameter to make the plot smaller
    ax.patch.set_alpha(1)
    background = "#ffffff"
    fig.set_facecolor(background)

    # Set the X-coordinate of the center of the pitch for each position
    positions_x = {'RB': 58, 'LB': 8, 'LW': 8, 'RW': 58, 'CDM': 33, 'CM': 48, 'AM': 18, 'CF': 33, 'CB': 33}  

    # Set the starting y-coordinate for each position
    start_y = {'RB': 38, 'LB': 38, 'LW': 80, 'RW': 80, 'CDM': 45, 'CM': 65, 'AM': 65, 'CF': 87, 'CB': 65}  

    # Annotate players for each position
    for position, players_data in zip(['RB', 'LB', 'LW', 'RW', 'CDM', 'CM', 'AM', 'CF'], [rb_players_data, lb_players_data, lw_players_data, rw_players_data, dm_players_data, cm_players_data, am_players_data, st_players_data]):
        for index, player in players_data.iterrows():
            ax.annotate(player['Player'], xy=(positions_x[position], start_y[position]), xytext=(positions_x[position], start_y[position]),
                        textcoords="offset points", ha='center', va='center', color='black', fontsize=6)
            start_y[position] -= 3  

    # Annotate left-footed CBs
    offset_left_cb = 0
    for index, player in left_cb_players_data.iterrows():
        ax.annotate(player['Player'], xy=(23, 30), xytext=(23, 30 + offset_left_cb),
                    textcoords="offset points", ha='center', va='center', color='black', fontsize=6)
        offset_left_cb -= 15  # Adjust the offset for left-footed CBs

    # Annotate right-footed CBs
    offset_right_cb = 0
    for index, player in right_cb_players_data.iterrows():
        ax.annotate(player['Player'], xy=(42, 30), xytext=(42, 30 + offset_right_cb),
                    textcoords="offset points", ha='center', va='center', color='black', fontsize=6)
        offset_right_cb -= 15  # Adjust the offset for right-footed CBs

    # Remove the red dot
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Create three columns with adjusted widths
    col1, col2, col3 = st.columns([0.8, 5, 0.8])

    # Display the pitch plot in the middle column
    with col2:
        st.pyplot(fig)

# Create the navigation menu in the sidebar
selected_tab = st.sidebar.radio("Navigation", ["Confidence Scores"])

# Based on the selected tab, display the corresponding content
if selected_tab == "Confidence Scores":
    display_data()
elif selected_tab == "Multi Player Comparison Tab":
    comparison_tab(df)
