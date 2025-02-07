import streamlit as st
import pandas as pd

st.title(' Machine learning App')

st.write('This app builds an machine learning model')

with st.expander('Data'):
  st.write('**Raw data**')
  df=pd.read_csv('https://raw.githubusercontent.com/Abhishekpn09/StreamlitProjects/refs/heads/master/.devcontainer/penguins_clean%20(1).csv')
  df

  st.write('**X**')
  X_raw=df.drop('species',axis=1)
  X_raw

  st.write('**Y**')
  Y_raw=df.species
  Y_raw

with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

with st.sidebar:
  st.header('Input features')
  island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
  gender = st.selectbox('Gender', ('male', 'female'))
  
  #Creating a Dataframe
data = {'island': island,
          'bill_length_mm': bill_length_mm,
          'bill_depth_mm': bill_depth_mm,
          'flipper_length_mm': flipper_length_mm,
          'body_mass_g': body_mass_g,
          'sex': gender}
input_df = pd.DataFrame(data, index=[0])
input_penguins = pd.concat([input_df, X_raw], axis=0)


encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)
input_row =df_penguins[:1]

target_mapper={'Adelie' :0 , 'Chinstrap' :1 ,'Gentoo':2}
def target_encoder(val):
  return target_mapper[val]

Y=Y_raw.apply(target_encode)
Y
Y_raw

with st.expander('Input features'):
  st.write('**Input penguin**')
  input_df
  st.write('**Combined penguins data**')
  input_penguins
  st.write('**Encoded input penguins**')
  input_row
