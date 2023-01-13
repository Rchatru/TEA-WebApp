import streamlit as st
import os
import s3fs
from functions import save2_s3
# import pandas as pd
# import numpy as np


st.set_page_config(
    page_title="ASD Check - Model",
    page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f440.png",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://github.com/Rchatru/TEA-WebApp/',
        'Report a bug': "https://github.com/Rchatru/TEA-WebApp/issues",
        'About': "# ASD WebApp. Roberto Chávez Trujillo."
    }
)

st.markdown('''
# 🎯 Models & Training

On this page, it is shown some of the training results of the classification algorithm.
It is also possible to provide new training data, if available, in order to re-train the model.
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
col1.metric("Individual 05c", "72.80%", "Control")
col2.metric("Individual 09c", "75.89%", "Control")
col3.metric("Individual 07p", "72.67%", "ASD",delta_color='inverse')
col4.metric("Individual 08p", "91.16%", "ASD",delta_color='inverse')

with st.sidebar.header('1. Upload your new training data'):
    input = st.sidebar.file_uploader("Upload your file", type=["csv"])

st.sidebar.subheader("Or")

with st.sidebar.header('2. Upload a new model file'):
    input = st.sidebar.file_uploader("Upload your file", type=[".json", ".bin", ".model"])

    successs = save2_s3(input, 'models/')
    if successs:
        st.sidebar.success('File saved successfully.', icon='✔')
    else:
        st.sidebar.error('Error saving file.', icon='❌')


st.image('static/Images/resultados.png',caption='Training Graphics', use_column_width=True)

col_1,col_2 = st.columns(2)
col_1.image('static/Images/resultados_txt.png',caption='Training, testing and validation metrics', use_column_width=True)
col_2.image('static/Images/confusion_func.png',caption='Normalised confusion matrix', use_column_width=True)
