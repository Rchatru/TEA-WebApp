import streamlit as st
import pandas as pd
import numpy as np
import pickle
# from xgboost import XGBClassifier
from functions import *


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
    df = pd.read_csv(input,sep=';')
    st.dataframe(df)

    # model = XGBClassifier()
    # model.load_model("XGBClassifier.json")

    # make predictions for test data
    # predictions = model.predict(df)


    csv = convert_df(df)
    with st.sidebar:
        st.header('2. Download results file')
        st.download_button('Download file', csv, 'results.csv', 'text/csv',key='download-csv')