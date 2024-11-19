import streamlit as st
# Configurar página
st.set_page_config(
    page_title="Landing Page",
    layout="centered",
)
# Estilo CSS para centrar el título
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
st.markdown('<div class="title">Prevenir accidentes cerebrovasculares es posible</div>', unsafe_allow_html=True)
st.write("Con este rápido cuestionario asegura de primera mano tu salud con nuestro modelo")
st.image('../data/background_image.jpeg', use_container_width= True)

if st.button("Haz un primer diagnóstico ahora"):
    st.query_params(page="app")
    st.experimental_rerun()

