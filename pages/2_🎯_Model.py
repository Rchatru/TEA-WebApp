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
# 🎯 Models & Training

En la presente página se pueden visualizar algunos de los resultados del 
entrenamiento del algoritmo de clasificación. También, se permite aportar nuevos 
datos de entremaniento en caso de disponer de ellos, para de este modo 
reentrenar el modelo.
 ''')

with st.sidebar.header('1. Upload your new training data'):
    input = st.sidebar.file_uploader("Upload your file", type=["csv"])