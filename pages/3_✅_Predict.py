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

if "predict_button" not in st.session_state:
    st.session_state.predict_button = False

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
        # st.session_state.man_test = 1
        df = upload_test_data()
            
        st.subheader('''
        Vista previa de los datos de entrada:
        ''')
        
        st.dataframe(df.head())

        with st.expander("See dataset debug info"):
            st.text(df_info(df))

        
        # Predicción
        if st.button('Predict !',key='button_test') or st.session_state.predict_button:
            st.session_state.predict_button = True

            st.markdown('''
            ## ✅ Resultados 
            
            A continuación, se muestra el dataset original junto a una nueva columna `Pred` que contiene la 
            predicción del modelo para cada una de las muestras individuales (filas).
            ''')
            pred = predict(df)
            # my_bar = st.progress(0)
            # for progress in range(100):
            #     time.sleep(0.01)
            #     my_bar.progress(progress + 1)
            # my_bar.empty()
            
            st.dataframe(pred)
            st.success('Prediction done!')

            st.markdown('''
            ### 📊 Predicciones Individuales 
            
            Finalmente, se detalla la clasificación a nivel de individuo. Por un lado, se indica el número de muestras disponibles por cada individuo, así 
            como la cantidad de ellas que han sido clasificadas como TEA y Control. Por otro lado, se tiene un deslizador que permite variar el umbral empleado 
            para determinar la clasificación de cada individuo. 
            ''')
            # Ahora se muestran los resultados

            col1,col2 = st.columns(2)
            col1.subheader('Predicciones individuales')
            col2.subheader('Umbral de clasificación')
      
            umbral = col2.slider('Ajuste el umbral de decisión', min_value=50, max_value=100, step=1)           
            

            cross_tab = crosstab(pred)
            
            col1.dataframe(cross_tab)


            unique_id = df.id.unique()
            col = st.columns(len(unique_id))
            for col,ind in zip(col,unique_id):
                percent,tipo,color = metrics(pred,ind,umbral)
                col.metric(label="Individio " + str(ind), value=percent, delta=tipo, delta_color=color)

            

            



            csv = convert_df(pred)
            with st.sidebar:
                st.header('2. Download results file')
                st.sidebar.caption('<p style="color:#484a55;">Descarga fichero procesado junto a predicción</p>', unsafe_allow_html=True)
                st.download_button('Download file', csv, 'results.csv', 'text/csv',key='download-csv')
