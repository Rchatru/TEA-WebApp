import streamlit as st
import pandas as pd
import os
import s3fs
# import numpy as np
# import io
# import pickle
# from xgboost import XGBClassifier
from functions import *
# import time


st.set_page_config(
    page_title="ASD Check - Predict",
    page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f440.png",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://github.com/Rchatru/TEA-WebApp/',
        'Report a bug': "https://github.com/Rchatru/TEA-WebApp/issues",
        'About': "# ASD WebApp. Roberto Chávez Trujillo."
    }
)

if "predict_button1" not in st.session_state:
    st.session_state.predict_button1 = False
if "predict_button2" not in st.session_state:
    st.session_state.predict_button2 = False

@st.cache_data
def load_print(input):
    df = pd.read_csv(input,sep=';')
    st.dataframe(df.head())
    return df

st.markdown('''
# ✅ Results & Predictions 
 
On this screen it is possible to check the prediction for a specific individual or groups, using the trained XGBoost model*.
''')

st.markdown('''
## Automated Data Processing:

Manual data preparation is no longer necessary. Background processes automatically match any *.csv* file containing eye-tracking data from Tobii TX-300
to the format required by the ML model. This includes removing unneeded features and applying necessary processing steps such as variable scaling and encoding.
However, the dataset format expected by the model is shown below for reference.
''')



st.table(
pd.DataFrame([[750,326,1,0,0],[0,0,0,1,0],[120,562,0,0,1]],
columns=['FixationPointX_(MCSpx)', 'FixationPointY_(MCSpx)', 'Fixation', 'Saccade', 'Unclassified'])
)

st.caption('''
*Note: Or any of the models desired by the user, by uploading the corresponding file to the Model page.
''')

 # Upload individual's data to be tested
with st.sidebar.header('1. Upload your data file'):
    input = st.sidebar.file_uploader("Upload your input file", type=["csv"])
    

if input is not None:  

    modelos,_ = show_s3_content('models/')
    default_model = modelos.index('XGBClassifier.bin')

    with st.sidebar:
        model_name = st.selectbox('Select a model to use in prediction',modelos,index=default_model)
    m = download_from_s3(model_name)

    st.subheader('''
    Vista previa de los datos de entrada:
    ''')

    df = load_print(input)

    with st.expander("See dataset debug info"):
        st.text(df_info(df))

    st.subheader('''
    Preview of the processed data:
    ''')
    new_df = check_df(df)
    st.dataframe(new_df.head())

    with st.expander("See dataset debug info"):
        st.text(df_info(new_df))

    # Predicción
    if st.button('Predict !',key='button_test1') or st.session_state.predict_button1:
        st.session_state.predict_button1 = True

        st.markdown('''
        ## ✅ Results 
        
        Below, the original dataset is shown together with a new `Pred` column containing the model
        prediction for each of the individual samples (rows).
        ''')
        pred = predict(new_df,m)
                    
        st.dataframe(pred)

        # DEBUG: eliminar al terminar
        # with st.expander("See dataset debug info"):
        #     st.text(df_info(pred))

        st.success('Prediction done!')

        st.markdown('''
        ### 📊 Individual Predictions 
        
        Finally, classification at the individual level is detailed. On one hand, it indicates
        the number of samples available for each individual, as well as the number of them that
        have been classified as ASD and Control. On the other hand, there is a slider that allows
        to vary the threshold used to determine the classification of each individual. 
        ''')
        # Ahora se muestran los resultados

        col1,col2 = st.columns(2)
        col1.subheader('Individual Predictions')
        col2.subheader('Classification Threshold')
    
        umbral = col2.slider('Adjust the decision threshold', min_value=50, max_value=100, step=1)           
        

        cross_tab = crosstab(pred)
        
        col1.dataframe(cross_tab)

        # Rejilla para la visualización de las métricas individuales
        unique_id = pred.id.unique()

        grid, row_num, col_num, max_col = make_grid(unique_id)

        for row in range (row_num):
            if row == row_num-1 and len(unique_id) % max_col != 0:
                col_num = len(unique_id) % max_col
                for col in range(col_num):
                    ind = unique_id[row*max_col+col]
                    percent,tipo,color = metrics(pred,ind,umbral)
                    grid[row][col].metric(label="Individual " + str(ind), value=percent, delta=tipo, delta_color=color)
            else:
                for col in range(col_num):
                    ind = unique_id[row*col_num+col]
                    percent,tipo,color = metrics(pred,ind,umbral)
                    grid[row][col].metric(label="Individual " + str(ind), value=percent, delta=tipo, delta_color=color)


        # col = st.columns(len(unique_id))
        # for col,ind in zip(col,unique_id):
        #     percent,tipo,color = metrics(pred,ind,umbral)
        #     col.metric(label="Individio " + str(ind), value=percent, delta=tipo, delta_color=color)

        

        
        if st.button('Refresh cache',help='Click to clear cache, double click to reload'):
            st.session_state.predict_button1 = False
            st.experimental_memo.clear()
    


        csv = convert_df(pred)
        with st.sidebar:
            st.header('2. Download results file')
            st.download_button('Download file', csv, 'results.csv', 'text/csv',key='download-csv')

else:
    with st.sidebar:
        placeholder = st.empty()
        # st.sidebar.caption('<p style="color:#484a55;">Cargar fichero con los individuos de test</p>', unsafe_allow_html=True)
        # st.sidebar.caption('Cargar fichero con los individuos de test', unsafe_allow_html=False)
        st.sidebar.markdown('<p style="font-size:14px;">Upload file with test individuals</p>', unsafe_allow_html=True)

    # NOTE: Se puede sustituir por un botón utilizando session_state
    if not st.sidebar.checkbox("Test Dataset"):
        placeholder.info("No file has been loaded. Select one or choose the available test dataset.")
    else:
        # st.session_state.man_test = 1
        df = upload_test_data()

        modelos,_ = show_s3_content('models/')

        with st.sidebar:
            model_name = st.selectbox('Select a model to use in prediction',modelos) 
        m = download_from_s3(model_name)   
        

        st.subheader('''
        Preview of input data:
        ''')

        st.markdown('''
        Now, in the drop-down menu in the left pane, *Select a model to use for prediction*, you can specify which of the stored models
        you want to use for inference.
        ''')
        
        st.dataframe(df.head())

        with st.expander("See dataset debug info"):
            st.text(df_info(df))

        
        # Predicción
        if st.button('Predict !',key='button_test2') or st.session_state.predict_button2:
            st.session_state.predict_button2 = True

            st.markdown('''
            ## ✅ Results 
            
            Below, the original dataset is shown together with a new `Pred` column containing the model
            prediction for each of the individual samples (rows).
            ''')
            pred = predict(df,m)
                        
            st.dataframe(pred)
            st.success('Prediction done!')

            st.markdown('''
            ### 📊 Individual Predictions 
            
            Finally, it displays the classification at the individual level. On one hand, it shows the number of samples available for each individual,
            along with the counts of samples classified as ASD and Control. On the other hand, there is a slider that allows you to adjust the threshold
            used for determining each individual's classification. 
            ''')
            # Ahora se muestran los resultados

            col1,col2 = st.columns(2)
            col1.subheader('Individual Predictions')
            col2.subheader('Classification Threshold')
      
            umbral = col2.slider('Adjust the decision threshold', min_value=50, max_value=100, step=1)           
            

            cross_tab = crosstab(pred)
            
            col1.dataframe(cross_tab)


            unique_id = df.id.unique()
            col = st.columns(len(unique_id))
            for col,ind in zip(col,unique_id):
                percent,tipo,color = metrics(pred,ind,umbral)
                col.metric(label="Individual " + str(ind), value=percent, delta=tipo, delta_color=color)

            

            
            if st.button('Refresh cache',help='Click to clear cache, double click to reload'):
                st.session_state.predict_button2 = False
                st.experimental_memo.clear()
                



            csv = convert_df(pred)
            with st.sidebar:
                st.header('2. Download results file')
                st.sidebar.caption('<p style="color:#484a55;">Download processed file together with prediction</p>', unsafe_allow_html=True)
                st.download_button('Download file', csv, 'results.csv', 'text/csv',key='download-csv')
