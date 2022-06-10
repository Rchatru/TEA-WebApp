import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
# from xgboost import XGBClassifier
from functions import *
import time


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
También se espera que el fichero de datos se encuentre estandarizado y codificado mediante One-Hot Encoding.
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

    print('antes boton')
    # Predicción
    if st.button('Predict !', help='Click to predict'):
        print('boton pulsado')
        pred = predict(df)

        my_bar = st.progress(0)
        for progress in range(100):
            time.sleep(0.01)
            my_bar.progress(progress + 1)
        my_bar.empty()
        st.write(pred)
        st.success('Prediction done!')
    
    csv = convert_df(df)
    with st.sidebar:
        st.header('2. Download results file')
        st.download_button('Download file', csv, 'results.csv', 'text/csv',key='download-csv')

else:
    with st.sidebar:
        placeholder = st.empty()
        st.sidebar.caption('<p style="color:#484a55;">Cargar fichero con los individuos de test</p>', unsafe_allow_html=True)

    # NOTE: No funciona el boton
    if not st.sidebar.checkbox("Test Dataset"):
        placeholder.info("No se ha cargado ningún fichero. Seleccione uno o escoja el dataset de test disponible.")
    else:
        st.session_state.man_test = 1
        df = upload_test_data()
            
        st.subheader('''
        Vista previa de los datos de entrada:
        ''')
        
        st.dataframe(df.head())

        with st.expander("See dataset debug info"):
            st.text(df_info(df))

        
        # Predicción
        pressed = st.button('Predict !',key='button_test')
        if pressed:
            pred = predict(df)
            my_bar = st.progress(0)
            for progress in range(100):
                time.sleep(0.01)
                my_bar.progress(progress + 1)
            my_bar.empty()
            st.write(pred)
            st.success('Prediction done!')

    


        csv = convert_df(df)
        with st.sidebar:
            st.header('2. Download results file')
            st.download_button('Download file', csv, 'results.csv', 'text/csv',key='download-csv')
