import streamlit as st
import pandas as pd
import pickle
import io
import sys

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

vars = ['FixationPointX_(MCSpx)','FixationPointY_(MCSpx)','Fixation','Saccade','Unclassified']

class stdScaler(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        self.means_ = X.mean(axis=0)
        self.std_dev_ = X.std(axis=0)
        return self

    def transform(self, X, y=None):
        return (X - self.means_[:X.shape[1]]) / self.std_dev_[:X.shape[1]]


def check_df(df_in):
   # En primer lugar ajusta el nombre de las columnas a la forma requerida
   df_in.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)
   cols = df_in.columns.tolist()

   # Compara los elementos de la lista de variables necesarias (vars) con las del archivo introducido (cols).
   if set(vars).issubset(set(cols)):
      if len(vars) == len(cols):
         df = df_in
      else:
         message = 'Las variables del archivo de entrada no coinciden con las esperadas. Se eliminarán las no necesarias.'
         st.info(message)
         df = df_in.drop(columns=set(cols) - set(vars))
   else:
      st.error("El archivo introducido no tiene todas las variables necesarias.")
      return False
      #sys.exit()
      
   df.fillna(0, inplace=True)

   # Cálculo de los cuartiles y rango IQR
   Q1 = df[vars].quantile(0.25)
   Q3 = df[vars].quantile(0.75)
   IQR = Q3 - Q1

   # Límites superior e inferior para el cálculo de los outliers
   k = 3
   l_sup = Q3 + k*IQR
   l_inf = Q1 - k*IQR

   # Se eliminan los outliers que se encuentren por encima del límite superior o por debajo del límite inferior
   fix_X = (encoded_train.loc[(encoded_train['FixationPointX_(MCSpx)'] >= l_sup['FixationPointX_(MCSpx)']) | (encoded_train['FixationPointX_(MCSpx)'] <= l_inf['FixationPointX_(MCSpx)'])])
   limp_train = encoded_train[~((encoded_train < m_inf) | (encoded_train > m_sup)).any(axis=1)]

   
   stdscaler = stdScaler()
   df[vars] = stdscaler.fit_transform(df[vars])

   return df

@st.cache
def predict(df):
   esc = 6
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