import streamlit as st
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
import pandas as pd
# import pickle
import io
import sys
import time

import xgboost


def convert_df(df):
   """
   Esta función permite convertir un dataframe en un archivo .csv listo para descargar.
   input:
      df: dataframe a convertir.
   output:
      archivo .csv
   """

   return df.to_csv().encode('utf-8')


def df_info(df):
   """
   Esta función captura el textI/O enviado directamente a consola por métodos como df.info().
   input:
      df: dataframe del que se desea mostrar los detalles.
   output:
      s: string con la información del dataframe.
   """
   # "Truco" necesario para mostrar cosas como df.info() en la página.
   buffer = io.StringIO()
   df.info(verbose=True,buf=buffer)
   s = buffer.getvalue() 
   return s

vars = ['FixationPointX_(MCSpx)','FixationPointY_(MCSpx)','Fixation','Saccade','Unclassified']

# class stdScaler(TransformerMixin, BaseEstimator):

#     def fit(self, X, y=None):
#         self.means_ = X.mean(axis=0)
#         self.std_dev_ = X.std(axis=0)
#         return self

#     def transform(self, X, y=None):
#         return (X - self.means_[:X.shape[1]]) / self.std_dev_[:X.shape[1]]



# La selección de escena debe estar aqui
# esc = 6
# X = df.loc[df['escena' + str(esc)] == 1]
# def check_df(df_in):
#    # En primer lugar ajusta el nombre de las columnas a la forma requerida
#    df_in.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)
#    cols = df_in.columns.tolist()

#    # Compara los elementos de la lista de variables necesarias (vars) con las del archivo introducido (cols).
#    if set(vars).issubset(set(cols)):
#       if len(vars) == len(cols):
#          df = df_in
#       else:
#          message = 'Las variables del archivo de entrada no coinciden con las esperadas. Se eliminarán las no necesarias.'
#          st.info(message)
#          df = df_in.drop(columns=set(cols) - set(vars))
#    else:
#       st.error("El archivo introducido no tiene todas las variables necesarias.")
#       return False
#       #sys.exit()
      
#    df.fillna(0, inplace=True)

#    # Cálculo de los cuartiles y rango IQR
#    Q1 = df[vars].quantile(0.25)
#    Q3 = df[vars].quantile(0.75)
#    IQR = Q3 - Q1

#    # Límites superior e inferior para el cálculo de los outliers
#    k = 3
#    l_sup = Q3 + k*IQR
#    l_inf = Q1 - k*IQR

#    # Se eliminan los outliers que se encuentren por encima del límite superior o por debajo del límite inferior
#    fix_X = (encoded_train.loc[(encoded_train['FixationPointX_(MCSpx)'] >= l_sup['FixationPointX_(MCSpx)']) | (encoded_train['FixationPointX_(MCSpx)'] <= l_inf['FixationPointX_(MCSpx)'])])
#    limp_train = encoded_train[~((encoded_train < m_inf) | (encoded_train > m_sup)).any(axis=1)]

   
#    stdscaler = stdScaler()
#    df[vars] = stdscaler.fit_transform(df[vars])

#    return df


# Es necesario añadir esta función al cache para que no se ejecute la predicción cada vez que se actualiza la página.
@st.experimental_memo(suppress_st_warning=True)
def predict(df):
   """
   Esta función permite realizar la clasificación de los datos en base al modelo XGBoost importado.
   input:
      df: dataframe con los datos a predecir. Se espera que el dataset esté previamente procesado con la función check_df() o
      importado con upload_test_data(). Debe contener las variables TEA e id. #TODO: #3 Hay que tener en cuenta que no siempre se podrá disponer de la variable objetivo, TEA.
   output:
      df_result: dataframe original con una nueva columna con la clase predicha, 'Pred'.
   """   
   
   X = df.loc[:, vars]
   Y = df.loc[:,['TEA', 'id']]

   model = xgboost.XGBClassifier()
   model.load_model('static/XGBClassifier.bin')
   # Importante para evitar bug de XGBoost https://github.com/dmlc/xgboost/issues/2073
   model._le = LabelEncoder().fit([0,1])

   result = model.predict(X)
   result = pd.DataFrame(result, columns=['Pred'])
   # Unir de nuevo los dataframes X, Y y result en uno solo
   df_result = pd.concat([X, Y, result], axis=1)
   
   # Se añade la barra de progreso en la función para evitar que se muestre cada vez que se actualiza la página.
   my_bar = st.progress(0)
   for progress in range(100):
      time.sleep(0.01)
      my_bar.progress(progress + 1)
   my_bar.empty()

   return df_result

def crosstab(df):
   """
   Función que muestra los resultados de la predicción.
   input: 
      df: Dataframe con los resultados de la predicción obtenido de la función predict
   output:
      crosstab: Dataframe con el número de muestras asignadas a cada clase según individuo.
   """
   
   # Para cada 'id' se muestra la cuenta de las muestras clasificadas como '0' y como '1'
   crosstab = pd.crosstab(df.id, df.Pred).rename(columns={0: 'Control', 1: 'TEA'})
   # Se añade columna con el total de muestras por individuo.
   crosstab['Muestras'] = crosstab['Control'] + crosstab['TEA']
   
   return crosstab



def metrics(df,ind,umbral):
   """
   Esta función permite realizar el cálculo del porcentaje de acierto para cada uno de los individuos y pasar este y otros datos auxiliares 
   al widget encargado de mostrar los resultados.
   input:
      df: dataframe obtenido de la función predict (debe contener las columnas 'id' y 'Pred').
      ind: individuo a calcular los resultados.
      umbral: umbral de confianza para la clasificación (obtenido del widget slider).
   output:
      percent: porcentaje de muestras sobre el total clasificadas correctamente.
      tipo: clase en la que se ha clasificado el individuo (TEA/Control).
      color: variable para el ajuste de color del widget dependiendo de la clase del individuo.
   """

   umbral = umbral/100
   total = df.loc[df['id'] == ind]['Pred'].count()
   sum1 = df.loc[df['id'] == ind]['Pred'].eq(1).sum()
   sum0 = df.loc[df['id'] == ind]['Pred'].eq(0).sum()
   if sum1 >= total*umbral:
      percent = '{:.2f}'.format(sum1/total*100) + '%'
      tipo = 'TEA'
      color = 'inverse'
   elif sum0 >= total*umbral:
      percent = '{:.2f}'.format(sum0/total*100) + '%'
      tipo = 'Control'
      color = 'normal'
   elif sum1 < total*umbral and sum0 < total*umbral and sum1 > sum0:
      percent = '{:.2f}'.format(sum1/total*100) + '%'
      tipo = 'Posible TEA'
      color = 'off'
   elif sum1 < total*umbral and sum0 < total*umbral and sum0 > sum1:
      percent = '{:.2f}'.format(sum0/total*100) + '%'
      tipo = 'Posible Control'
      color = 'off'
   else:
      percent = 0
      tipo = 'No definido'
      color = 'off'

   return percent,tipo,color



def upload_test_data():
   """
   Esta función permite subir un archivo de testeo para realizar la clasificación. Se ha escogido un dataset ya procesado en el 
   archivo Principal_sript.py (One Hot Encoding, FillNa, outliers, estandarización, etc.)
   input:
      None
   output:
      df: dataframe con los datos de test.
   """

   # Open stored .csv file at static folder an convert to dataframe
   df = pd.read_csv("static/test_data.csv",sep=';') 
   cols = df.columns.to_list()
   # Es necesario disponer una variable que contenga todos los id de los individuos
   df['id'] = pd.get_dummies(df[['07p', '08p', '05c', '09c']]).idxmax(1)
   # Se eliminan todas las demás menos id y TEA
   df = df.drop(columns=set(cols) - set(vars+['id','TEA']))
   return df
