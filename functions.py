import streamlit as st
import pickle

@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')

@st.cache
def predict(df):
   esc = 6
   BestVars = ['FixationPointX_(MCSpx)','FixationPointY_(MCSpx)','Fixation','Saccade','Unclassified']
   X = df.loc[df['escena' + str(esc)] == 1]
   X = X.loc[:, BestVars]
   model = pickle.load(open('/static/XGBClassifier.sav', 'rb'))
   result = model.predict(X)

   return result


# model = XGBClassifier()
# model.load_model("XGBClassifier.json")

# esc = 6
# BestVars = ['FixationPointX_(MCSpx)','FixationPointY_(MCSpx)','Fixation','Saccade','Unclassified']
# fichero normalizado (std)
# validacion = model.predict(X_val)