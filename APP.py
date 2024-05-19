from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Deteccion de Plagas en la agricultura Mexicana",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.header("Configuración del modelo de aprendizaje automático")

# Model Options
model_types_available = ['Deteccion', 'OtraTarea', 'OtraTarea2']  # Agrega más tareas según sea necesario
model_type = st.sidebar.multiselect("Seleccionar tarea", model_types_available, default=['Deteccion'])

confidence = float(st.sidebar.slider(
    "Seleccione la confianza del modelo", 25, 100, 40)) / 100

if not model_type:
    model_type = ['Deteccion']

selected_task = model_type[0]

if selected_task == 'Deteccion':
    model_path = Path(settings.DETECTION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Imagen/Config")

source_img = st.sidebar.file_uploader(
    "Elige una imagen...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

if source_img:
    uploaded_image = PIL.Image.open(source_img)
    st.image(uploaded_image, caption="Imagen Cargada", use_column_width=True)

    if st.sidebar.button('Detectar Objeto'):
        res = model.predict(uploaded_image, conf=confidence)
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        st.image(res_plotted, caption='Detected Image', use_column_width=True)
        
        with st.expander("Resultados de la detección"):
            for box in boxes:
                st.write(box.data)
