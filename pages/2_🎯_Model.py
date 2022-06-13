import streamlit as st
# import pandas as pd
# import numpy as np

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

# Hack para quitar las flechas del widget
st.write(
    """
    <style>
    [data-testid="stMetricDelta"] svg {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Individuo 05c", "72.80%", "Control")
col2.metric("Individuo 09c", "75.89%", "Control")
col3.metric("Individuo 07p", "72.67%", "TEA",delta_color='inverse')
col4.metric("Individuo 08p", "91.16%", "TEA",delta_color='inverse')

with st.sidebar.header('1. Upload your new training data'):
    input = st.sidebar.file_uploader("Upload your file", type=["csv"])

st.sidebar.subheader("Or")

with st.sidebar.header('2. Upload a new model file'):
    input = st.sidebar.file_uploader("Upload your file", type=[".json", ".bin", ".model"])


st.image('static/Images/resultados.png',caption='Gráficas de entrenamiento', use_column_width=True)
st.image('static/Images/confusion_func.png',caption='Matriz de confusión normalizada', use_column_width=True)