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

st.markdown('''
Aunque la página se encarga de adecuar los datos a la forma requerida, se recomienda que el usuario introduzca los datos de entrada en 
el formato CSV siguiendo el siguiente esquema de columnas: 
esc = 6
BestVars = ['FixationPointX_(MCSpx)','FixationPointY_(MCSpx)','Fixation','Saccade','Unclassified']
fichero normalizado (std)
validacion = model.predict(X_val)
''')

st.table(
pd.DataFrame(10*np.random.randn(3, 5),
columns=['A', 'B', 'C', 'D', 'E'])
)



 # Upload individual's data to be tested
with st.sidebar.header('1. Upload your data file'):
    input = st.sidebar.file_uploader("Upload your input file", type=["csv"])


if input is not None:  

    st.markdown('''
    Vista previa de los datos de entrada:
    ''')
    df = pd.read_csv(input,sep=';')
    st.dataframe(df.head())

    



    csv = convert_df(df)
    with st.sidebar:
        st.header('2. Download results file')
        st.download_button('Download file', csv, 'results.csv', 'text/csv',key='download-csv')