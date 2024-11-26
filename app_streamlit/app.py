import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import os

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

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "test.csv"))
model = pickle.load(open('final_model_rf.pkl','rb'))
# Streamlit 

st.logo(
    os.path.join(os.path.dirname(__file__), "medicine_logo.png"), size= 'large'
    #link="httpshttp://localhost:8502/#predecir-si-una-persona-puede-sufrir-de-accidentes-cerebrovascularesstreamlit.io/gallery",
)

tab1, tab2 = st.tabs(['Caso de uso', 'Funcionamiento del modelo'])

with tab1:
    if "current_view" not in st.session_state:
        st.session_state.current_view = "main" 

    if st.session_state.current_view == "main":
        st.markdown('<div class="title">Prevenir accidentes cerebrovasculares es posible</div>', unsafe_allow_html=True)
        st.image('background_image.jpeg', use_container_width= True)      
        if st.button("Haz un primer diagnóstico ahora"):
            switch_view("diagnosis")

    elif st.session_state.current_view == "diagnosis":
        if st.button("Volver"):
            switch_view("main")

        st.title('Asegura tu salud. Haz este rápido test para saber si te encuentras en riesgo')
        st.image(os.path.join(os.path.dirname(__file__), "dataset-cover.jpg"), use_container_width=True)

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

        #Tiene depressive disorder o no 
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

            # Convertir el diccionario a un DataFrame para el modelo
            df_parametros = pd.DataFrame([params])

            #st.write('Esto es lo que le pasaremos al modelo para hacer la predicción')
            #st.dataframe(df_parametros, use_container_width=True)
            st.subheader("Resultado: ")
            # Mostrar el resultado de la predicción
            prob = float(model.predict_proba(df_parametros)[0][1])
            prob_str = str(round(prob * 100, 2)) + "%"

            if prob <= 0.2:
                mensaje = "Su evaluación indica un riesgo muy bajo de accidente cerebrovascular. Le recomendamos mantener sus hábitos saludables y realizar chequeos médicos de rutina."
                st.success(f"Probabilidad de riesgo: {prob_str}. \n\n{mensaje}",icon="✅")
            elif prob > 0.2 and prob <= 0.4:
                mensaje = "Tiene un riesgo bajo de accidente cerebrovascular. Sin embargo, es importante seguir cuidando su salud y considerar un chequeo médico regular para mantenerse en óptimas condiciones."
                st.success(f"Probabilidad de riesgo: {prob_str}. \n\n{mensaje}",icon="✅")
            elif prob > 0.4 and prob <= 0.6:
                mensaje = "Su evaluación muestra un riesgo moderado de accidente cerebrovascular. Le sugerimos concertar una cita médica para una evaluación más detallada y recibir orientación preventiva."
                st.warning(f"Probabilidad de riesgo: {prob_str}. \n\n{mensaje}", icon="🚨")
            elif prob > 0.6 and prob <= 0.8:
                mensaje = "Se ha detectado un riesgo alto de accidente cerebrovascular. Es importante que programe una consulta médica lo antes posible para analizar su situación y planificar medidas preventivas."
                st.warning(f"Probabilidad de riesgo: {prob_str}. \n\n{mensaje}", icon="🚨")
            else:
                mensaje = "Tiene un riesgo muy alto de accidente cerebrovascular según nuestra evaluación. Por favor, busque atención médica de inmediato para recibir un diagnóstico y tratamiento adecuado."
                st.warning(f"Probabilidad de riesgo: {prob_str}. \n\n{mensaje}", icon="🚨")

with tab2:
    X = df[['AgeGroup', 'HadHeartAttack', 'diabetesGroup','SmokerGroup', 'AlcoholDrinkers', 'HadArthritis','HadKidneyDisease', 'HadDepressiveDisorder']]
    y = df['HadStroke']
    y_pred = model.predict(X)

    st.write('''
        ## Modelo usado
        Esta app utiliza un modelo de machine learning supervisado de clasificación binaria, concretamente un RandomForestClassifier, el cual 
        construye árboles de decisión a partir de diferentes muestras y toma su voto mayoritario para decidir la clasificación 
        y el promedio en caso de regresión.
            ''')
    st.code("RandomForestClassifier(n_estimators= 200,min_samples_split=4,max_depth=6,criterion='gini',class_weight='balanced', random_state=42)")
    st.write("## Caso de uso")
    st.write("Personas que tengan dudas sobre su estado actual y que crean que puedan ser propensas a tener un accidente cerebrovascular, y mediante esta rápida evaluación, haciendo uso de un modelo de machine learning de clasificación binaria, pueda saber si estan corriendo riesgo de sufrir este tipo de accidentes y trabajar con su equipo de atención médica y evaluar estos riesgos, que la mayoría son prevenibles o tratables.")       
    
    st.write("Métricas:")
    # Cálculo de métricas
    accuracy = round(metrics.accuracy_score(y, y_pred) * 100, 2)
    precision = round(metrics.precision_score(y, y_pred) * 100, 2)
    recall = round(metrics.recall_score(y, y_pred) * 100, 2)
    st.markdown(f"**Exactitud:** <span style='color:green'>{accuracy}%</span>", unsafe_allow_html=True)
    st.markdown(f"**Precisión:** <span style='color:green'>{precision}%</span>", unsafe_allow_html=True)
    st.markdown(f"**Exhaustividad:** <span style='color:green'>{recall}%</span>", unsafe_allow_html=True)
    st.markdown("7 de cada 10 pacientes estan en riesgo de sufrir accidentes cerebrovasculares, aunque solo 5 de 10 serán casos reales")

    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    st.write(f'''
        ## Parametros
        Este modelo toma en cuenta las siguientes carácteristicas del paciente, con su importancia para el modelo y ver cuales son más interesantes para el mismo:
        * **Categoría de edad o {feat_importances.index[0]}**: {round(feat_importances[0], 2)}%.
        * **Cardiopatía o {feat_importances.index[1]}**: {round(feat_importances[1], 2)}%.
        * **Categoría de diabetes o {feat_importances.index[2]}**: {round(feat_importances[2], 2)}%.
        * **Categoría de fumador o {feat_importances.index[3]}**: {round(feat_importances[3], 2)}%.
        * **Si bebe alcohol o {feat_importances.index[4]}**: {round(feat_importances[4], 2)}%.
        * **Si sufre de artritis o {feat_importances.index[5]}**: {round(feat_importances[5], 2)}%.
        * **Si sufre de nefropatía o {feat_importances.index[6]}**: {round(feat_importances[6], 2)}%.
        * **Si sufre de depresión o {feat_importances.index[7]}**: {round(feat_importances[7], 2)}%.
             
        ***Muchas condiciones médicas comunes pueden aumentar la probabilidad de tener un accidente cerebrovascular***
             ''')
    
        