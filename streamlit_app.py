import streamlit as st
import pandas as pd

st.title(' Machine learning App')

st.write('This app builds an machine learning model')

df=pd.read_csv('https://raw.githubusercontent.com/Abhishekpn09/StreamlitProjects/refs/heads/master/.devcontainer/penguins_clean%20(1).csv')

