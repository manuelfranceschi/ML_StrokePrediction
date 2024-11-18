# Librerias
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn.metrics as metrics

# Dataframe y modelo 
df = pd.read_csv('../data/test/test.csv')
model = pickle.load(open('../models/trained_model_2.pkl','rb'))

#1era idea: Fondo de un paciente hablando con un medico, titulo 'La prevención de accidentes cerebrovasculares pueden ser sencillas de evitar', 
#un boton de 'haz un primer diagnostico ahora', un footer de aviso 'este cuestionario no deberia tomarse como una recomendación médica más allá
#de una primera aproximación de si estas en riesgo de sufrir este tipo de accidentes.
st.title("Los accidentes cerebrovasculares son prevenibles y tratables")
st.subheader("Con este rápido cuestionario asegura de primera mano tu salud con nuestro modelo")
st.image('../data/background_image.jpeg', use_container_width= True)
app_button = st.button('haz tu diagnostico ahora', use_container_width = True)

if app_button:
    st.write("funciona la interacción")
