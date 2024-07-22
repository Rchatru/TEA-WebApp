import streamlit as st
import functions as fc

# Icon 👀 alojado en: https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f440.png
# Ha sido necesario cambiar la ubicación del mismo ya que streamlit busca por defecto en maxcdn que ha sido cerrado
# https://github.com/twitter/twemoji/issues/580

st.set_page_config(
    page_title="ASD Check - Home",
    page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f440.png",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://github.com/Rchatru/TEA-WebApp/',
        'Report a bug': "https://github.com/Rchatru/TEA-WebApp/issues",
        'About': "# ASD WebApp. Roberto Chávez Trujillo."
    }
)

st.title('🔎 ASD-check WebApp')

st.markdown('''
This web application, developed using Python and Streamlit, is designed to offer a user-friendly interface for interacting with the results of PhD research focused on diagnosing Autism Spectrum Disorder (ASD) using eye-tracking data.

The application is organized into several sections, each with a distinct purpose:

1. [Home](https://asd-check.streamlit.app/): This introductory page provides an overview and allows access to the other sections.
2. [Models & Training](https://asd-check.streamlit.app/Model): Here, users can provide additional training data to re-train the algorithm (under construction) and manage different trained models.
3. [Results & Predictions](https://asd-check.streamlit.app/Predict): This section enables users to input data for a specific individual to obtain a diagnosis from the trained model.
''')


