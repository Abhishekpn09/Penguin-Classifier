import streamlit as st
import pandas as pd

st.title(' Machine learning App')

st.write('This app builds an machine learning model')

with st.expander('Data'):
  st.write('**Raw data**')
  df=pd.read_csv('https://raw.githubusercontent.com/Abhishekpn09/StreamlitProjects/refs/heads/master/.devcontainer/penguins_clean%20(1).csv')
  df

  st.write('**X**')
  X=df.drop('species',axis=1)
  X

  st.write('**Y**')
  Y=df.species
  Y


  with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')
  
  
