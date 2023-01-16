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

st.title('🏠 ASD WebApp')

st.markdown('''
WebApp built with Python and Streamlit designed to provide a more user-friendly interaction with the results of the PhD research
aimed at the diagnosis of Autism Spectrum Disorder using eye-tracker eye dynamics data of the participants.

The application is divided into different sub-pages, each with a distinct purpose:

- First of all, there is the "Home" screen, which contains a basic introduction and from which the rest of the screens can be accessed.
- Secondly, there is a [page](https://asd-check.streamlit.app/Model) where it is possible to provide further training data to carry out a re-training of the algorithm (under construction).
- Finally, in the [third](https://asd-check.streamlit.app/Predict) section, it is possible to enter the data of a specific individual in order to carry out a diagnosis by the model.
''')


