import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn.metrics as metrics

# Configurar página y CSS
st.set_page_config(
    page_title="Prevención de Accidentes Cerebrovasculares",
    layout="centered",
)
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 2em;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Funciones, modelos y dataframes
def switch_view(view_name):
    st.session_state.current_view = view_name
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
df = pd.read_csv('../data/test/test.csv')
model = pickle.load(open('../models/trained_model_2.pkl','rb'))
# Streamlit 
if "current_view" not in st.session_state:
    st.session_state.current_view = "main" 

if st.session_state.current_view == "main":
    st.markdown('<div class="title">Prevenir accidentes cerebrovasculares es posible</div>', unsafe_allow_html=True)
    st.image('../data/background_image.jpeg', use_container_width= True)      
    if st.button("Haz un primer diagnóstico ahora"):
        switch_view("diagnosis")

elif st.session_state.current_view == "diagnosis":
    if st.button("Volver"):
        switch_view("main")


    st.title('Predecir si una persona puede sufrir de accidentes cerebrovasculares')
    st.image("../data/dataset-cover.jpg", use_container_width=True)

    # Edad
    age_input = st.text_input("Introduce tu edad:", placeholder= '22')
    if age_input.isdigit():
        st.write("El número es:", int(age_input))
    else:
        st.write("Por favor, introduce un número válido.")

    # cardiopatía 0:1
    opt_heart_attack = {"Si": 1, "No": 0}
    heart_attack = st.radio(
        "¿Has tenido alguna cardiopatía?",
        opt_heart_attack.keys(),
    )

    # Tiene diabetes - varios rangos
    opt_diabetes = {
        "No": 0,
        "Si, pero solo durante el embarazo (mujeres)": 1,
        "Prediabetes": 2,
        "Si": 3
    }
    diabetes = st.radio(
        "¿Tienes diabetes?",
        opt_diabetes.keys()
    )

    # Es fumador - varios rangos
    opt_smoker= {
        "Nunca he fumado": 0,
        "Soy exfumador": 1,
        "Fumo ocasionalmente (algunos días)": 2,
        "Fumo diariamente": 3
    }
    smoker = st.radio(
        "¿Eres fumador?",
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
        "¿Sufres de artritis?",
        opt_arthritis.keys()
    )

    # Tiene kidney disease 0:1
    opt_kidney_disease = {"Si": 1, "No": 0}
    kidney_disease = st.radio(
        "¿Sufres de nefropatía (enfermedad renal)?",
        opt_kidney_disease.keys()
    )

    opt_depressive_disorder = {"Si": 1, "No": 0}
    dep_disorder = st.radio(
        "¿Has sufrido o sufres de depresión?",
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

        st.write('Esto es lo que le pasaremos al modelo para hacer la predicción')
        st.dataframe(df_parametros, use_container_width=True)

        # Mostrar el resultado de la predicción
        pred = model.predict(df_parametros)

        st.subheader("Resultado: ")
        if pred == 0:
            st.success("No eres propenso a sufrir accidentes cerebrovasculares, por lo que puedes estar tranquilo!",icon="✅")
        else:
            st.warning('Tienes indicios de poder sufrir un accidente cerebrovascular, te recomendamos acudas a tu médico de cabecera', icon="🚨")

        st.write('Métricas de test')

        X = df[['AgeGroup', 'HadHeartAttack', 'diabetesGroup','SmokerGroup', 'AlcoholDrinkers', 'HadArthritis','HadKidneyDisease', 'HadDepressiveDisorder']]
        y = df['HadStroke']
        y_pred = model.predict(X)

        df_test = pd.DataFrame({"Real": y, 
                            "Pred": y_pred})
        st.dataframe(df_test)

        st.write(f'accuracy_score', metrics.accuracy_score(y,y_pred))
        st.write(f'precision_score', metrics.precision_score(y,y_pred))
        st.write(f'recall_score', metrics.recall_score(y,y_pred))

    """
    Feedback: 
    añadir una sección de 'Quieres saber como funciona este modelo?'
    2. Nombre del modelo: lo mismo, llamar al usuario a usarlo. ver ejemplos de landingpage
    de pandas o de numpy, pyplot etc.
    3. Métricas del test: bien, pero explicarlas en vez de enseñarlas. En el propio FAQ traducir
    las metricas, por ejemplo con el recall 'El 60% de los pacientes te los predecirá bien'
    4. Resultados: Utilizar el predict_proba para añadir varias posibilidades de predicción.
    Ejemplo: 20% de probabilidad: Eres poco propenso a sufrir de este tipo de accidentes
    Ejemplo: 40% de probabilidad: No eres propenso a sufrir este tipo de accidentes, aunque 
    recomendamos acudas a tu equipo médico para comprobarlo.
    5. Propuesta de valor: la herramienta va dedicada a páginas web de clinicas, hospitales y
    aseguradoras, incitando a la consulta médica siempre y cuando haya probabilidad de accidente
    Otro modelo de redes neuronales para predecir si el usuario puede tener accidentes neuronales
    orientado a predicciones médicas que ayuden al médico a tomar una decisión

    1era idea: Fondo de un paciente hablando con un medico, titulo 'La prevención de accidentes cerebrovasculares pueden ser sencillas de evitar', 
    un boton de 'haz un primer diagnostico ahora', un footer de aviso 'este cuestionario no deberia tomarse como una recomendación médica más allá
    de una primera aproximación de si estas en riesgo de sufrir este tipo de accidentes.
    """

