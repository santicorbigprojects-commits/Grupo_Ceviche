import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Productividad RIKOS", layout="wide")

st.title("Calculadora de Productividad – RIKOS")

st.write("Ingrese los datos operativos para estimar productividad en sala y cocina.")

venta_diaria = st.number_input(
    "Venta diaria total (€)",
    min_value=0.0,
    step=100.0
)

local = st.selectbox(
    "Local",
    ["MERIDIANA", "CAN VIDALET", "BADAL", "CORNELLA",
     "GLORIES", "SANTA COLOMA", "ICARIA", "LLURIA"]
)

share_glovo = st.slider(
    "Porcentaje de ventas por Glovo",
    min_value=0.0,
    max_value=1.0,
    step=0.01
)

tipo_productividad = st.selectbox(
    "Tipo de productividad",
    ["ESPERADO", "MÁXIMO", "MÍNIMO"]
)

if st.button("Calcular"):
    
    venta_glovo = venta_diaria * share_glovo
    venta_sala = venta_diaria - venta_glovo

    st.subheader("Distribución de Ventas")

    ventas_df = pd.DataFrame({
        "Concepto": ["Venta total", "Venta sala", "Venta Glovo"],
        "Monto (€)": [venta_diaria, venta_sala, venta_glovo]
    })

    ventas_df["Monto (€)"] = ventas_df["Monto (€)"].round(2)
    st.table(ventas_df)


    st.success("App conectada correctamente 🚀")
