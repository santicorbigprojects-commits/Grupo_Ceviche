import streamlit as st
import pandas as pd

# -----------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------
st.set_page_config(
    page_title="Calculadora Operativa – Grupo Ceviche",
    layout="wide"
)

st.title("🍽️ Calculadora Operativa – Grupo Ceviche")
st.caption("Simulador de ventas y productividad (sala y cocina)")

# -----------------------------
# INPUTS
# -----------------------------
st.sidebar.header("🔢 Inputs")

venta_diaria = st.sidebar.number_input(
    "Venta diaria total (€)",
    min_value=0.0,
    value=3000.0,
    step=100.0
)

local = st.sidebar.selectbox(
    "Local",
    ["Ceviche 103", "Otro local"]
)

porcentaje_glovo = st.sidebar.number_input(
    "% de venta por Glovo (0 a 1)",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05
)

# Inputs operativos
st.sidebar.header("👥 Recursos")

mozos = st.sidebar.number_input(
    "Cantidad de mozos",
    min_value=1,
    value=4
)

cocineros = st.sidebar.number_input(
    "Cantidad de cocineros",
    min_value=1,
    value=3
)

horas_operacion = st.sidebar.number_input(
    "Horas de operación del día",
    min_value=1,
    value=10
)

ticket_promedio = st.sidebar.number_input(
    "Ticket promedio (€)",
    min_value=1.0,
    value=25.0
)

# -----------------------------
# CÁLCULOS DE VENTAS
# -----------------------------
venta_glovo = venta_diaria * porcentaje_glovo
venta_sala = venta_diaria - venta_glovo

ventas_df = pd.DataFrame({
    "Concepto": ["Venta total", "Venta en sala", "Venta en Glovo"],
    "Monto (€)": [venta_diaria, venta_sala, venta_glovo]
})

# -----------------------------
# PRODUCTIVIDAD SALA
# -----------------------------
tickets_totales = venta_diaria / ticket_promedio
tickets_sala = venta_sala / ticket_promedio

productividad_sala_df = pd.DataFrame({
    "Indicador": [
        "Tickets totales",
        "Tickets en sala",
        "Tickets por mozo",
        "Ventas por mozo (€)",
        "Ventas por mozo por hora (€)"
    ],
    "Valor": [
        tickets_totales,
        tickets_sala,
        tickets_sala / mozos,
        venta_sala / mozos,
        (venta_sala / mozos) / horas_operacion
    ]
})

# -----------------------------
# PRODUCTIVIDAD COCINA
# -----------------------------
productividad_cocina_df = pd.DataFrame({
    "Indicador": [
        "Ventas por cocinero (€)",
        "Ventas por cocinero por hora (€)",
        "Tickets por cocinero",
        "Tickets por cocinero por hora"
    ],
    "Valor": [
        venta_diaria / cocineros,
        (venta_diaria / cocineros) / horas_operacion,
        tickets_totales / cocineros,
        (tickets_totales / cocineros) / horas_operacion
    ]
})

# -----------------------------
# OUTPUTS
# -----------------------------
st.header("📊 Resultados de Ventas")
st.dataframe(
    ventas_df.round(2),
    use_container_width=True
)

st.header("🧍 Productividad de Sala")
st.dataframe(
    productividad_sala_df.round(2),
    use_container_width=True
)

st.header("👨‍🍳 Productividad de Cocina")
st.dataframe(
    productividad_cocina_df.round(2),
    use_container_width=True
)
