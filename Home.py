import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
     page_title="ASD WebApp",
     page_icon="👀",
     menu_items={
         'Get Help': 'https://github.com/Rchatru/TEA-WebApp/',
         'Report a bug': "https://github.com/Rchatru/TEA-WebApp/issues",
         'About': '''ASD WebApp.
         https://github.com/Rchatru/         
         Roberto Chávez Trujillo.'''
     }
 )

st.title('ASD WebApp')

st.markdown('''
WebApp built with Python and Streamlit designed to provide a more user-friendly interaction with the results of the PhD research
aimed at the diagnosis of Autism Spectrum Disorder using eye-tracker eye dynamics data of the participants.

The application is divided into different sub-pages, each with a distinct purpose:

- First of all, there is the "Home" screen, which contains a basic introduction and from which the rest of the screens can be accessed.
- Secondly, there is a [page](https://share.streamlit.io/rchatru/tea-webapp/Home.py/Model) where it is possible to provide further training data to carry out a re-training of the algorithm (under construction).
- Finally, in the [third](https://share.streamlit.io/rchatru/tea-webapp/Home.py/Predict) section, it is possible to enter the data of a specific individual in order to carry out a diagnosis by the model.
''')
