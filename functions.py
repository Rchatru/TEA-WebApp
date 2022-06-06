import streamlit as st
import pandas as pd
import pickle
import io

@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')

@st.cache
def df_info(df):
   # For showing dataframe info
   buffer = io.StringIO()
   df.info(verbose=True,buf=buffer)
   s = buffer.getvalue() 
   return s

@st.cache
def predict(df):
   esc = 6
   vars = ['FixationPointX_(MCSpx)','FixationPointY_(MCSpx)','Fixation','Saccade','Unclassified']
   X = df.loc[df['escena' + str(esc)] == 1]
   X = X.loc[:, vars]
   model = pickle.load(open('static/XGBClassifier.sav', 'rb'))
   result = model.predict(X)
   return result

@st.cache
def upload_test_data():
   # Open stored .csv file at static folder an convert to dataframe
   df = pd.read_csv("static/test_data.csv",sep=';') 
   return df

# model = XGBClassifier()
# model.load_model("XGBClassifier.json")

# esc = 6
# BestVars = ['FixationPointX_(MCSpx)','FixationPointY_(MCSpx)','Fixation','Saccade','Unclassified']
# fichero normalizado (std)
# validacion = model.predict(X_val)