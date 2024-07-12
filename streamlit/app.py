import os
import streamlit as st

# Set page configuration at the very beginning
st.set_page_config(page_title='DinoDetector', page_icon='🦖', layout='wide')

# Print the current working directory
current_directory = os.getcwd()
st.write(f"Current working directory: {current_directory}")
