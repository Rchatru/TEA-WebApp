import streamlit as st
import os
import s3fs
from functions import save2_s3
from functions import show_file_structure
from functions import show_s3_content
from functions import delete_from_s3
from datetime import datetime
import time
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

On this page, you can view detailed training results of the classification algorithm, including performance metrics and visualizations such as loss
and precision-recall curves.

In addition, this section also allows you to upload new training data to re-train the model, enabling continuous improvement and adaptation
to new data *(work in progress)*. You can also manage different trained models by uploading pre-trained files or exporting the current one.
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
    
    if input is not None:
        date = datetime.today().strftime('%d-%m-%Y_%H-%M')
        successs = save2_s3(input, f"models/model_{date}.bin")
        if successs:
            st.sidebar.success('File saved successfully.', icon='✔')
        else:
            st.sidebar.error('Error saving file.', icon='❌')


st.image('static/Images/resultados.png',caption='Training Graphics', use_column_width=True)

col_1,col_2 = st.columns(2)
col_1.image('static/Images/resultados_txt.png',caption='Training, testing and validation metrics', use_column_width=True)
col_2.image('static/Images/confusion_func.png',caption='Normalised confusion matrix', use_column_width=True)

st.markdown('''## ⚙ Manage Stored Models
Here, the user can upload a new trained model using the menu available in the sidebar. The uploaded model will be securely stored in the corresponding
AWS bucket and will be used on the Predict page for inference. Upon upload, the model file is renamed according to the current server date and time to
ensure unique identification.

Additionally, a file tree of available models is displayed, providing an overview of all stored models. This feature allows users to manage them
efficiently, including the option to remove obsolete ones.
''')

text = show_file_structure('models/')

with st.expander('Show file structure',expanded=True):
    # st.text(text)
    st.code(text)

st.markdown('''### Choose the model to delete''')

modelos,_ = show_s3_content('models/')
eliminar = st.multiselect('Select the model/s to delete',modelos)
# modelos.insert(0,'<Select a file>')
# eliminar = st.selectbox('Select the model to delete',modelos,index=0)



# if eliminar is not '<Select a file>':
if eliminar:
    st.warning(f'Are you sure you want to delete the model/s: *{[el for el in eliminar]}*, (specify pin)?')
    password = st.text_input("Enter a password", type="password")
    if st.button('Delete') and password == st.secrets["PIN"]:
        my_bar = st.progress(0)
        for progress in range(100):
            time.sleep(0.01)
            my_bar.progress(progress + 1)
        my_bar.empty()
        for el in eliminar:
            correcto = delete_from_s3(el)
        if correcto:
            st.success(f'Model/s *{eliminar}* deleted successfully.')
            text = show_file_structure('models/')

            with st.expander('Remaining files after deletion',expanded=True):
                # st.text(text)
                st.code(text)
   
