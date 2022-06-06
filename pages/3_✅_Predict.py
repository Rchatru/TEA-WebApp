import streamlit as st
import pandas as pd
import numpy as np
import io
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
''')



st.table(
pd.DataFrame([[750,326,1,0,0],[0,0,0,1,0],[120,562,0,0,1]],
columns=['FixationPointX_(MCSpx)', 'FixationPointY_(MCSpx)', 'Fixation', 'Saccade', 'Unclassified'])
)

st.caption('''
Nota: El modelo actual desarrollado sólo precisa los datos de la escena 6 y se basa únicamente en las variables 
`['FixationPointX_(MCSpx)','FixationPointY_(MCSpx)','Fixation','Saccade','Unclassified']`.
También se espera que el fichero de datos se encuentre normalizado y codificado mediante One-Hot Encoding.
''')

 # Upload individual's data to be tested
with st.sidebar.header('1. Upload your data file'):
    input = st.sidebar.file_uploader("Upload your input file", type=["csv"])
    


if input is not None:  

    st.subheader('''
    Vista previa de los datos de entrada:
    ''')
    df = pd.read_csv(input,sep=';')
    st.dataframe(df.head())

    with st.expander("See dataset debug info"):
        st.text(df_info(df))
    
    csv = convert_df(df)
    with st.sidebar:
        st.header('2. Download results file')
        st.download_button('Download file', csv, 'results.csv', 'text/csv',key='download-csv')

else:
    with st.sidebar:
        placeholder = st.empty()
        st.sidebar.caption('<p style="color:#484a55;">Cargar fichero con los individuos de test</p>', unsafe_allow_html=True)
    if not st.sidebar.button("Test Dataset"):
        placeholder.info("No se ha cargado ningún fichero. Seleccione uno o escoja el dataset de test disponible.")
    else:
        df = upload_test_data()
            
        st.subheader('''
        Vista previa de los datos de entrada:
        ''')
        
        st.dataframe(df.head())

        with st.expander("See dataset debug info"):
            st.text(df_info(df))
        
        csv = convert_df(df)
        with st.sidebar:
            st.header('2. Download results file')
            st.download_button('Download file', csv, 'results.csv', 'text/csv',key='download-csv')
