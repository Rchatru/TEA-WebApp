import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
     page_title="TEA WebApp",
     page_icon="👀",
     menu_items={
         'Get Help': 'https://github.com/Rchatru/TEA-WebApp/',
         'Report a bug': "https://github.com/Rchatru/TEA-WebApp/issues",
         'About': "# TEA WebApp. Roberto Chávez Trujillo."
     }
 )

st.markdown('''
# ✅ Results & Predictions 
 
En esta pantalla se puede consultar la predicción para un individuo o grupos en concreto que efectúa el modelo XGBoost entrenado.
''')

 # Upload individual's data to be tested
with st.sidebar.header('1. Upload your data file'):
    input = st.sidebar.file_uploader("Upload your input file", type=["csv"])

if input is not None:
    

    out =3
    with st.sidebar.header('2. Download results file'):
        st.download_button('Download file', out)