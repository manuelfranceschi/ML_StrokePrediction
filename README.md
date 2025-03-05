![Empresas de videojuegos](./data/header.png)
# Accidentes cerebrovasculares
Modelo de machine learning

Este proyecto tiene como objetivo predecir el riesgo de accidente cerebrovascular en asegurados utilizando técnicas de machine learning sobre datos de salud. La predicción temprana del riesgo de accidente cerebrovascular es crucial para implementar medidas preventivas, mejorar los resultados de los pacientes y optimizar los recursos de salud.

[Puedes probarlo aquí](https://mlstrokeprediction-manuelfranceschi.streamlit.app/)

## Propósito
La predicción del riesgo de accidente cerebrovascular ayuda a las aseguradoras a:

1. Prevenir y gestionar factores de riesgo: Se pueden implementar intervenciones personalizadas para pacientes de alto riesgo, enfocándose en cambios de estilo de vida, ajustes de medicación y monitoreo regular.
2. Optimizar la asignación de recursos: Identificar pacientes de alto riesgo permite a los sistemas de salud priorizar recursos de manera eficiente, reduciendo costos a largo plazo asociados con el tratamiento y la rehabilitación.

## Características Principales

* Análisis Exploratorio de Datos (EDA): Visualizaciones y estadísticas descriptivas para comprender el conjunto de datos y métricas de evaluación de los modelos.
* Preprocesamiento de Datos: Codificación de variables categóricas.
* Modelado Predictivo: Implementación de algoritmos como Regresión Logística, RandomForest, XGBoost, GradientBoost, etc.
* Demo: Interfaz interactiva desarrollada con Streamlit para facilitar la predicción en tiempo real.

## Datos
El conjunto de datos incluye variables como edad, hipertensión, enfermedades cardíacas, estado de tabaquismo e IMC, las cuales están asociadas con el riesgo de accidente cerebrovascular.

## Uso
Al ejecutar este modelo, los usuarios de páginas de seguros médicos y clínicas podrán tener un rápido diagnostico si estan en riesgo de sufrir este tipo de accidentes o no, el cual arrojará una recomendación en base al porcentaje de predicción.

### Librerías Necesarias

* Pandas
* NumPy
* Matplotlib y Seaborn
* Scikit-learn
* XGBoost
* Streamlit2

