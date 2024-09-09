import streamlit as st # type: ignore
import pandas as pd # type: ignore
import joblib
import numpy as np
import xgboost as xgb

# Inyectar CSS personalizado para los estilos
BACKGROUND_COLOR = '#0D1B2A'
LETRAS = '#A9D6E5'
COLOR_BUTTOM = '#415A77'
COLOR_LOADFILE = '#1B263B'
        
st.markdown(
    """
    <style>
    body {
        background-color: #042433; /* Azul oscuro */
        color: #A9D6E5; /* Celeste */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #6391D2;
    }
    .stButton button {
        background-color: #1B263B; /* Azul más oscuro */
        color: #A9D6E5; /* Celeste */
        border-radius: 10px;
        border: 2px solid #A9D6E5;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #415A77; /* Azul intermedio */
    }
    .css-1dp5vir {
        background-color: #1B263B !important; /* Panel de carga de archivo */
    }
    .stFileUploader label {
        color: #A9D6E5 !important; /* Texto del cargador de archivos */
    }
    /* Color del control deslizante */
    .css-1d391kg .stSlider>div>input[type="range"]::-webkit-slider-thumb {
        background-color: #A9D6E5; /* Cambia el color del control deslizante */
    }
    /* Color de la pista del slider */
    .css-1d391kg .stSlider>div>input[type="range"]::-webkit-slider-runnable-track {
        background-color: #A9D6E5; /* Cambia el color de la pista del slider */
    }
    /* Color del área de selección del slider */
    .css-1d391kg .stSlider>div>input[type="range"]::-webkit-slider-runnable-track {
        background: linear-gradient(to right, #00bfff 0%, #00bfff 50%, #004080 50%, #004080 100%);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Título de la aplicación
st.title("Predicción de Presupuestos con IA")

# Instrucciones para el usuario
st.write("Subir un archivo CSV con las características de las viviendas para predecir el presupuesto.")

# Cargar el archivo CSV
#uploaded_file = st.file_uploader("Subir un archivo CSV", type=["csv"])

# Front 
m2 = float(st.text_input("Metros Cuadrados", value="0.0"))

tipo_intervencion = int(st.checkbox('Tipo de intervención'))
demolicion_previa = int(st.checkbox('¿Húbo demolición previa?'))
tipo_de_obra = int(st.checkbox('Tipo de obra'))
piscina = int(st.checkbox('Tendrá piscina?'))
jardin = int(st.checkbox('¿Tendrá jardin?'))
aparcamiento_sub = int(st.checkbox('¿Tendrá aparcamientos subterraneos?'))
aparcamiento_exterior = int(st.checkbox('¿Tendrá aparcamientos exterior?'))

antiguedad = int(st.slider('Antiguedad :', 1820, 2023, 1990))
num_viviendas = int(st.slider('Número de viviendas :',0 ,200 ,1))
num_plantas = int(st.slider('Número de plantas :',0 ,20 ,1))
num_plantas_sub = int(st.slider('Número de plantas subterráneas :',0 ,3 ,0))
num_locales_comer = int(st.slider('Número de locales comerciales :',0 ,5 ,0))
num_trasteros = int(st.slider('Número de trasteros :',0 ,80 ,0))
num_aparcamientos_sub = int(st.slider('Número de aparcamientos :',0 ,555 ,0))
calidad_acabados = int(st.slider('Calidad de acabados :',0 ,2 ,0))
cod_municipio = int(st.slider('Código de municipio :',1 ,23 ,1))

# Creamos el array de entrada
X_list =    [m2, 
              tipo_intervencion,
              demolicion_previa,
              tipo_de_obra,
              piscina,
              jardin,
              aparcamiento_sub,
              aparcamiento_exterior,
              antiguedad,
              num_viviendas,
              num_plantas,
              num_plantas_sub,
              num_locales_comer,
              num_trasteros,
              num_aparcamientos_sub,
              calidad_acabados,
              cod_municipio
              ]

#X = np.array([float(elemento) for elemento in X_list])
X = np.array(X_list, dtype=np.float64)
X = X.reshape(1,-1)

# Botón para ejecutar el modelo
if st.button("Predecir"):
    if len(X) > 0:
        
        # Cargar el modelo y los parámetros de normalización guardados
        scaler = joblib.load('scaler.pkl')
        model = joblib.load('xgboost_model.pkl')
        
        # Mostrar las primeras filas del DataFrame cargado
        st.write("Datos cargados:")
        st.write(X)
        
        data_scaled = scaler.transform(X)
        
        # Realizar predicciones con el modelo XGBoost
        predicciones = model.predict(data_scaled)
        
        # Mostrar las predicciones
        st.write("Predicciones del presupuesto estimado:")
        st.write(predicciones)

