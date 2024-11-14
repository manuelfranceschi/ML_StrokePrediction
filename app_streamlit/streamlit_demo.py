# Librerias
import streamlit as st
import pickle
import pandas as pd
import numpy as np
# Dataframe y modelo 
df = pd.read_csv('../data/test/test.csv')
model = pickle.load(open('../models/finished_model.pkl','rb'))

#Titulo e imagen
st.image("../data/dataset-cover.jpg")
st.title('Predecir si una persona puede sufrir de accidentes cerebrovasculares')
st.markdown("""
            ***Primera aproximación del modelo***
            """)
# Variables y funciones
def age_to_category(age_str): 
    age = int(age_input)
    if age >= 18 and age <= 39:
        age_category = 1
    elif age >= 40 and age <= 59:
        age_category = 2
    elif age >= 60 and age <= 79:
        age_category = 3
    elif age >= 80:
        age_category = 4
    return age_category
# Botones y demás
# Edad
age_input = st.text_input("Introduce tu edad:", placeholder= '22')

if age_input.isdigit():
    st.write("El número es:", int(age_input))

else:
    st.write("Por favor, introduce un número válido.")

# Ataques al corazón alguna vez 0:1
opt_heart_attack = {"Si": 1, "No": 0}
heart_attack = st.radio(
    "Has tenido ataques al corazón?",
    opt_heart_attack.keys(),
)

# Tiene diabetes - varios rangos
opt_diabetes = {
    "No": 0,
    "Yes, but only during pregnancy (female)": 1,
    "No, pre-diabetes or borderline diabetes": 2,
    "Yes": 3
}
diabetes = st.radio(
    "Tienes diabetes?",
    opt_diabetes.keys()
)

# Es fumador - varios rangos
opt_smoker= {
    "Never smoked": 0,
    "Former smoker": 1,
    "Current smoker - now smokes some days": 2,
    "Current smoker - now smokes every day": 3
}
smoker = st.radio(
    "Eres fumador?",
    opt_smoker.keys()
)

# beber alcohol 0:1 
opt_drinker = {"Si": 1, "No": 0}
drinker = st.radio(
    "¿Bebes alcohol?",
    opt_drinker.keys()
)

# Tiene arthritis 0:1
opt_arthritis = {"Si": 1, "No": 0}
arthritis = st.radio(
    "¿Has tenido artritis?",
    opt_arthritis.keys()
)

# Tiene kidney disease 0:1
opt_kidney_disease = {"Si": 1, "No": 0}
kidney_disease = st.radio(
    "¿Has sufrido de alguna enfermedad del colón?",
    opt_kidney_disease.keys()
)

opt_depressive_disorder = {"Si": 1, "No": 0}
dep_disorder = st.radio(
    "¿Has sufrido de depresión?",
    opt_depressive_disorder.keys()
)
if st.button("Realizar Predicción"):
    # Recoger las variables en un diccionario
    params = {
        "AgeGroup": age_to_category(age_input),
        "HadHeartAttack": opt_heart_attack[heart_attack],
        "diabetesGroup": opt_diabetes[diabetes],
        "SmokerGroup": opt_smoker[smoker],
        "AlcoholDrinkers": opt_drinker[drinker],
        "HadArthritis": opt_arthritis[arthritis],
        "HadKidneyDisease": opt_kidney_disease[kidney_disease],
        "HadDepressiveDisorder": opt_depressive_disorder[dep_disorder]
    }

    # Convertir el diccionario a un DataFrame de una fila para el modelo
    df_parametros = pd.DataFrame([params])

    st.write('Esto es lo que le pasaremos al modelo para trabajar')
    st.dataframe(df_parametros, use_container_width=True)

    # Mostrar el resultado de la predicción
    pred = model.predict(df_parametros)
    st.subheader("Resultado: ")
    if pred == 0:
        st.success("No eres propenso a sufrir accidentes cerebrovasculares, por lo que puedes estar tranquilo!",icon="✅")
    else:
        st.error('Tienes indicios de poder sufrir un accidente cerebrovascular, te recomendamos acudas a tu médico de cabecera', icon="🚨")
