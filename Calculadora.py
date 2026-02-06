import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --------------------------------------------------
# CONFIGURACIÓN STREAMLIT
# --------------------------------------------------
st.set_page_config(
    page_title="Calculadora de Productividad – RIKOS",
    layout="wide"
)

st.title("Calculadora de Eficiencia y Horas – RIKOS")

# --------------------------------------------------
# INPUTS
# --------------------------------------------------
local = st.selectbox(
    "Selecciona el local",
    ["MERIDIANA", "CAN VIDALET", "BADAL", "CORNELLA",
     "GLORIES", "SANTA COLOMA", "ICARIA", "LLURIA"]
)

tipo_productividad = st.selectbox(
    "Tipo de productividad",
    ["ESPERADO", "MÁXIMO", "MÍNIMO"]
)

venta_diaria = st.number_input(
    "Venta diaria total (€)",
    min_value=0.0,
    value=3000.0,
    step=100.0
)

share_glovo = st.number_input(
    "% de ventas en Glovo (0 a 1)",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.01
)

# --------------------------------------------------
# HORARIOS DE APERTURA Y CIERRE
# --------------------------------------------------
horarios_locales = {
    "LLURIA": {
        "LUNES": {"abre": "12:30", "cierra": "23:30"},
        "MARTES": {"abre": "12:30", "cierra": "23:30"},
        "MIÉRCOLES": {"abre": "12:30", "cierra": "23:30"},
        "JUEVES": {"abre": "12:30", "cierra": "23:30"},
        "VIERNES": {"abre": "12:30", "cierra": "23:30"},
        "SÁBADO": {"abre": "9:00", "cierra": "23:59"},
        "DOMINGO": {"abre": "9:00", "cierra": "23:59"}
    },
    "ICARIA": {
        "LUNES": {"abre": "13:00", "cierra": "23:00"},
        "MARTES": {"abre": "13:00", "cierra": "23:00"},
        "MIÉRCOLES": {"abre": "13:00", "cierra": "23:00"},
        "JUEVES": {"abre": "13:00", "cierra": "23:00"},
        "VIERNES": {"abre": "13:00", "cierra": "23:00"},
        "SÁBADO": {"abre": "13:00", "cierra": "23:00"},
        "DOMINGO": {"abre": "13:00", "cierra": "23:00"}
    },
    "BADAL": {
        "LUNES": {"abre": "13:00", "cierra": "23:30"},
        "MARTES": {"abre": "13:00", "cierra": "23:30"},
        "MIÉRCOLES": {"abre": "13:00", "cierra": "23:30"},
        "JUEVES": {"abre": "13:00", "cierra": "23:30"},
        "VIERNES": {"abre": "13:00", "cierra": "24:30"},
        "SÁBADO": {"abre": "13:00", "cierra": "24:30"},
        "DOMINGO": {"abre": "13:00", "cierra": "23:30"}
    },
    "SANTA COLOMA": {
        "LUNES": {"abre": "13:00", "cierra": "23:00"},
        "MARTES": {"abre": "13:00", "cierra": "23:00"},
        "MIÉRCOLES": {"abre": "13:00", "cierra": "23:00"},
        "JUEVES": {"abre": "13:00", "cierra": "23:00"},
        "VIERNES": {"abre": "13:00", "cierra": "23:00"},
        "SÁBADO": {"abre": "13:00", "cierra": "23:00"},
        "DOMINGO": {"abre": "13:00", "cierra": "23:00"}
    },
    "CORNELLA": {
        "LUNES": {"abre": "12:30", "cierra": "23:00"},
        "MARTES": {"abre": "12:30", "cierra": "23:00"},
        "MIÉRCOLES": {"abre": "12:30", "cierra": "23:00"},
        "JUEVES": {"abre": "12:30", "cierra": "23:00"},
        "VIERNES": {"abre": "12:30", "cierra": "23:00"},
        "SÁBADO": {"abre": "12:30", "cierra": "23:00"},
        "DOMINGO": {"abre": "12:30", "cierra": "23:00"}
    },
    "CAN VIDALET": {
        "LUNES": {"abre": "12:30", "cierra": "23:15"},
        "MARTES": {"abre": "12:30", "cierra": "23:15"},
        "MIÉRCOLES": {"abre": "12:30", "cierra": "23:15"},
        "JUEVES": {"abre": "12:30", "cierra": "23:15"},
        "VIERNES": {"abre": "12:30", "cierra": "23:15"},
        "SÁBADO": {"abre": "12:30", "cierra": "23:15"},
        "DOMINGO": {"abre": "12:30", "cierra": "23:15"}
    },
    "GLORIES": {
        "LUNES": {"abre": "13:00", "cierra": "22:30"},
        "MARTES": {"abre": "13:00", "cierra": "22:30"},
        "MIÉRCOLES": {"abre": "13:00", "cierra": "22:30"},
        "JUEVES": {"abre": "13:00", "cierra": "22:30"},
        "VIERNES": {"abre": "13:00", "cierra": "22:30"},
        "SÁBADO": {"abre": "13:00", "cierra": "22:30"},
        "DOMINGO": {"abre": "13:00", "cierra": "22:30"}
    },
    "MERIDIANA": {
        "LUNES": {"abre": "13:00", "cierra": "23:00"},
        "MARTES": {"abre": "13:00", "cierra": "23:00"},
        "MIÉRCOLES": {"abre": "13:00", "cierra": "23:00"},
        "JUEVES": {"abre": "13:00", "cierra": "23:00"},
        "VIERNES": {"abre": "13:00", "cierra": "23:00"},
        "SÁBADO": {"abre": "13:00", "cierra": "23:00"},
        "DOMINGO": {"abre": "13:00", "cierra": "23:00"}
    }
}

# --------------------------------------------------
# CÁLCULO DE VENTAS
# --------------------------------------------------
venta_glovo = venta_diaria * share_glovo
venta_sala = venta_diaria - venta_glovo

# --------------------------------------------------
# TABLA DE VENTAS
# --------------------------------------------------
ventas_df = pd.DataFrame({
    "Concepto": ["Venta total", "Venta sala", "Venta Glovo"],
    "Monto (€)": [venta_diaria, venta_sala, venta_glovo]
})

# --------------------------------------------------
# PARÁMETROS DE LOS MODELOS
# --------------------------------------------------
dias = ["LUNES", "MARTES", "MIÉRCOLES", "JUEVES", "VIERNES", "SÁBADO", "DOMINGO"]

coef_dia_sala = {
    "LUNES": 18.2192,
    "MARTES": 18.2916,
    "MIÉRCOLES": 10.8842,
    "JUEVES": 18.4559,
    "VIERNES": 7.4385,
    "SÁBADO": 1.5453,
    "DOMINGO": 0
}

coef_dia_cocina = {
    "LUNES": 6.1412,
    "MARTES": 4.2809,
    "MIÉRCOLES": 3.5314,
    "JUEVES": 7.7640,
    "VIERNES": 1.7950,
    "SÁBADO": -3.2005,
    "DOMINGO": 0
}

coef_local_sala = {
    "CAN VIDALET": 47.5779,
    "CORNELLA": 32.0810,
    "GLORIES": 30.6683,
    "ICARIA": 26.5032,
    "LLURIA": -17.7891,
    "MERIDIANA": 23.1716,
    "SANTA COLOMA": 28.7016,
    "BADAL": 0
}

coef_local_cocina = {
    "CAN VIDALET": 32.2937,
    "CORNELLA": 46.5149,
    "GLORIES": 48.7297,
    "ICARIA": 49.4127,
    "LLURIA": -17.3252,
    "MERIDIANA": 10.9482,
    "SANTA COLOMA": 38.5073,
    "BADAL": 0
}

share_glovo_promedio = {
    "MERIDIANA": 0.203457008,
    "CAN VIDALET": 0.164354398,
    "BADAL": 0.442485954,
    "CORNELLA": 0.375005705,
    "GLORIES": 0.442987607,
    "SANTA COLOMA": 0.527390956,
    "ICARIA": 0.327043339,
    "LLURIA": 0.323162578
}

# --------------------------------------------------
# COMPONENTES NUMÉRICOS
# --------------------------------------------------
ln_ventas_sala = np.log(venta_sala) if venta_sala > 0 else 0
ln_ventas_total = np.log(venta_diaria) if venta_diaria > 0 else 0
share_glovo_centrado = share_glovo - share_glovo_promedio[local]

ajuste_sala = 14.72
ajuste_cocina = 11.91

if tipo_productividad == "MÁXIMO":
    factor_sala = ajuste_sala
    factor_cocina = ajuste_cocina
elif tipo_productividad == "MÍNIMO":
    factor_sala = -ajuste_sala
    factor_cocina = -ajuste_cocina
else:
    factor_sala = 0
    factor_cocina = 0

# --------------------------------------------------
# CÁLCULO PRODUCTIVIDAD AJUSTADA (para cálculo interno)
# --------------------------------------------------
prod_sala = []

for d in dias:
    valor = (
        -426.3496 +
        coef_dia_sala[d] +
        64.4162 * ln_ventas_sala +
        coef_local_sala.get(local, 0) +
        factor_sala
    )
    prod_sala.append(valor)

tabla_sala = pd.DataFrame(
    [prod_sala],
    index=[f"PRODUCTIVIDAD {tipo_productividad}"],
    columns=dias
)

prod_cocina = []

for d in dias:
    valor = (
        -565.6808 +
        coef_dia_cocina[d] +
        80.0758 * ln_ventas_total +
        16.8018 * share_glovo_centrado +
        coef_local_cocina.get(local, 0) +
        factor_cocina
    )
    prod_cocina.append(valor)

tabla_cocina = pd.DataFrame(
    [prod_cocina],
    index=[f"PRODUCTIVIDAD {tipo_productividad}"],
    columns=dias
)

# --------------------------------------------------
# CÁLCULO HORAS TEÓRICAS
# --------------------------------------------------
tabla_horas_sala = venta_sala / tabla_sala
tabla_horas_cocina = venta_diaria / tabla_cocina

tabla_horas_sala.index = [f"HORAS {tipo_productividad}"]
tabla_horas_cocina.index = [f"HORAS {tipo_productividad}"]

# --------------------------------------------------
# FUNCIONES AUXILIARES
# --------------------------------------------------
def convertir_hora_a_minutos(hora_str):
    """Convierte hora HH:MM a minutos desde medianoche, manejando horas >= 24"""
    partes = hora_str.split(':')
    horas = int(partes[0])
    minutos = int(partes[1])
    return horas * 60 + minutos

def minutos_a_bloque(minutos):
    """Convierte minutos a formato de bloque HH:MM"""
    horas = minutos // 60
    mins = minutos % 60
    return f"{horas}:{mins:02d}"

def ordenar_bloques_horarios(bloques):
    """Ordena bloques: 8:00-23:30 primero, luego 0:00-1:30"""
    bloques_dt = pd.to_datetime(bloques, format='%H:%M')
    bloques_dia = []
    bloques_noche = []
    
    for i, bloque in enumerate(bloques):
        hora = bloques_dt[i].hour
        minuto = bloques_dt[i].minute
        
        if 2 <= hora < 8:  # Excluir 2:00-7:59
            continue
        elif hora >= 8:  # 8:00-23:59
            bloques_dia.append(bloque)
        elif hora <= 1:  # 0:00-1:59
            bloques_noche.append(bloque)
    
    bloques_dia_sorted = sorted(bloques_dia, key=lambda x: pd.to_datetime(x, format='%H:%M'))
    bloques_noche_sorted = sorted(bloques_noche, key=lambda x: pd.to_datetime(x, format='%H:%M'))
    
    return bloques_dia_sorted + bloques_noche_sorted

# --------------------------------------------------
# CARGAR Y PROCESAR DISTRIBUCIONES
# --------------------------------------------------
@st.cache_data
def cargar_distribucion_ventas():
    try:
        df = pd.read_csv('data/distribucion_ventas_local.csv', sep=';')
    except:
        try:
            df = pd.read_csv('data/distribucion_ventas_local.csv', sep='\t')
        except:
            df = pd.read_csv('data/distribucion_ventas_local.csv', sep=',')
    
    df['bloque_30min'] = pd.to_datetime(df['bloque_30min'], format='%H:%M:%S').dt.strftime('%H:%M')
    df.columns = df.columns.str.strip()
    
    if 'dia' in df.columns:
        df = df.rename(columns={'dia': 'día'})
    
    df['porcentaje_ventas'] = pd.to_numeric(df['porcentaje_ventas'], errors='coerce').fillna(0)
    
    df['día'] = df['día'].str.strip().str.upper()
    df['día'] = df['día'].replace({
        'MIERCOLES': 'MIÉRCOLES',
        'SABADO': 'SÁBADO'
    })
    
    if 'Distribución' in df.columns:
        df['Distribución'] = df['Distribución'].str.strip()
    elif 'Distribucion' in df.columns:
        df = df.rename(columns={'Distribucion': 'Distribución'})
        df['Distribución'] = df['Distribución'].str.strip()
    
    return df

# Cargar datos
try:
    df_distribucion = cargar_distribucion_ventas()
    df_local = df_distribucion[df_distribucion['local'] == local].copy()
    
    if len(df_local) == 0:
        st.warning(f"⚠️ No hay datos de distribución disponibles para {local}")
        st.stop()
    
    # Preparar datos SALA
    df_sala = df_local[df_local['Distribución'] == "local"].copy()
    horas_semanales_sala = tabla_horas_sala.sum(axis=1).values[0]
    df_sala['horas_bloque'] = df_sala['porcentaje_ventas'] * horas_semanales_sala
    df_sala['hora_num'] = pd.to_datetime(df_sala['bloque_30min'], format='%H:%M').dt.hour
    df_sala = df_sala[~((df_sala['hora_num'] >= 2) & (df_sala['hora_num'] < 8))].copy()
    
    matriz_horas_sala = df_sala.pivot_table(
        index='bloque_30min',
        columns='día',
        values='horas_bloque',
        fill_value=0
    )
    
    # Preparar datos COCINA
    df_cocina = df_local[df_local['Distribución'] == "glovo&local"].copy()
    horas_semanales_cocina = tabla_horas_cocina.sum(axis=1).values[0]
    df_cocina['horas_bloque'] = df_cocina['porcentaje_ventas'] * horas_semanales_cocina
    df_cocina['hora_num'] = pd.to_datetime(df_cocina['bloque_30min'], format='%H:%M').dt.hour
    df_cocina = df_cocina[~((df_cocina['hora_num'] >= 2) & (df_cocina['hora_num'] < 8))].copy()
    
    matriz_horas_cocina = df_cocina.pivot_table(
        index='bloque_30min',
        columns='día',
        values='horas_bloque',
        fill_value=0
    )
    
    # Preparar datos VENTAS
    df_ventas = df_local[df_local['Distribución'] == "glovo&local"].copy()
    venta_semanal = venta_diaria * 7
    df_ventas['venta_bloque'] = df_ventas['porcentaje_ventas'] * venta_semanal
    df_ventas['hora_num'] = pd.to_datetime(df_ventas['bloque_30min'], format='%H:%M').dt.hour
    df_ventas = df_ventas[~((df_ventas['hora_num'] >= 2) & (df_ventas['hora_num'] < 8))].copy()
    
    matriz_ventas = df_ventas.pivot_table(
        index='bloque_30min',
        columns='día',
        values='venta_bloque',
        fill_value=0
    )
    
    # Ordenar días en todas las matrices
    dias_orden = ["LUNES", "MARTES", "MIÉRCOLES", "JUEVES", "VIERNES", "SÁBADO", "DOMINGO"]
    matriz_horas_sala = matriz_horas_sala.reindex(columns=dias_orden, fill_value=0)
    matriz_horas_cocina = matriz_horas_cocina.reindex(columns=dias_orden, fill_value=0)
    matriz_ventas = matriz_ventas.reindex(columns=dias_orden, fill_value=0)
    
    # Reordenar bloques en todas las matrices
    bloques_ordenados = ordenar_bloques_horarios(matriz_horas_sala.index.tolist())
    matriz_horas_sala = matriz_horas_sala.reindex(bloques_ordenados)
    matriz_horas_cocina = matriz_horas_cocina.reindex(bloques_ordenados)
    
    bloques_ordenados_ventas = ordenar_bloques_horarios(matriz_ventas.index.tolist())
    matriz_ventas = matriz_ventas.reindex(bloques_ordenados_ventas)
    
    # Calcular acumulados por día
    horas_sala_por_dia = matriz_horas_sala.sum(axis=0)
    horas_cocina_por_dia = matriz_horas_cocina.sum(axis=0)
    ventas_por_dia = matriz_ventas.sum(axis=0)
    horas_totales_por_dia = horas_sala_por_dia + horas_cocina_por_dia
    productividad_efectiva_por_dia = ventas_por_dia / horas_totales_por_dia
    
except FileNotFoundError:
    st.error("⚠️ Archivo no encontrado. Asegúrate de que data/distribucion_ventas_local.csv existe.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error al cargar datos: {str(e)}")
    st.stop()

# --------------------------------------------------
# CALCULAR PERSONAL REQUERIDO CON RESTRICCIONES
# --------------------------------------------------
def calcular_personal_requerido(matriz_horas, area, local, dias_orden):
    """
    Calcula el personal requerido redondeando hacia arriba y aplicando restricciones de horarios
    
    Parámetros:
    - matriz_horas: DataFrame con horas requeridas por bloque
    - area: 'SALA' o 'COCINA'
    - local: nombre del local
    - dias_orden: lista de días de la semana
    
    Retorna:
    - matriz_personal: DataFrame con cantidad de personal por bloque
    """
    # Crear copia de la matriz y redondear hacia arriba
    matriz_personal = matriz_horas.copy()
    matriz_personal = np.ceil(matriz_personal).astype(int)
    
    # Aplicar restricciones por horarios
    horarios = horarios_locales[local]
    
    for dia in dias_orden:
        if dia not in horarios:
            continue
            
        hora_abre = horarios[dia]["abre"]
        hora_cierra = horarios[dia]["cierra"]
        
        minutos_abre = convertir_hora_a_minutos(hora_abre)
        minutos_cierra = convertir_hora_a_minutos(hora_cierra)
        
        # Determinar cantidad de personal según área y local
        if area == "SALA":
            personal_apertura = 2 if local == "LLURIA" else 1
            bloques_antes = 1  # 1 bloque de 30 min antes
            personal_cierre = 2
            bloques_despues = 1  # 1 bloque después
        else:  # COCINA
            personal_apertura = 2
            bloques_antes = 2  # 2 bloques de 30 min antes
            personal_cierre = 2
            bloques_despues = 1  # 1 bloque después
        
        # Aplicar personal antes de apertura
        for i in range(bloques_antes):
            minutos_bloque = minutos_abre - (bloques_antes - i) * 30
            bloque_str = minutos_a_bloque(minutos_bloque)
            
            if bloque_str in matriz_personal.index:
                # Tomar el máximo entre el valor actual y el personal de apertura
                matriz_personal.loc[bloque_str, dia] = max(
                    matriz_personal.loc[bloque_str, dia],
                    personal_apertura
                )
        
        # Aplicar personal después de cierre
        for i in range(1, bloques_despues + 1):
            minutos_bloque = minutos_cierra + i * 30
            bloque_str = minutos_a_bloque(minutos_bloque)
            
            if bloque_str in matriz_personal.index:
                # Tomar el máximo entre el valor actual y el personal de cierre
                matriz_personal.loc[bloque_str, dia] = max(
                    matriz_personal.loc[bloque_str, dia],
                    personal_cierre
                )
    
    return matriz_personal

# Calcular matrices de personal
matriz_personal_sala = calcular_personal_requerido(matriz_horas_sala, "SALA", local, dias_orden)
matriz_personal_cocina = calcular_personal_requerido(matriz_horas_cocina, "COCINA", local, dias_orden)

# Calcular horas reales (suma de trabajadores en bloques / 2)
horas_reales_sala = matriz_personal_sala.sum(axis=0) / 2
horas_reales_cocina = matriz_personal_cocina.sum(axis=0) / 2
horas_reales_totales = horas_reales_sala + horas_reales_cocina

# Calcular productividad efectiva real
productividad_efectiva_real = ventas_por_dia / horas_reales_totales

# --------------------------------------------------
# OUTPUTS - TABLAS PRINCIPALES
# --------------------------------------------------
st.header("💰 Ventas diarias")
st.dataframe(ventas_df.round(2), use_container_width=True)

# --------------------------------------------------
# 1. PRODUCTIVIDAD EFECTIVA (TEÓRICA)
# --------------------------------------------------
st.markdown("---")
st.header("💼 Productividad Efectiva (Teórica)")
st.info("**Productividad Efectiva** = Ventas del día / (Horas Sala + Horas Cocina)")

tabla_prod_efectiva = pd.DataFrame({
    "Métrica": ["Productividad Efectiva (€/h)"],
    **{dia: [productividad_efectiva_por_dia[dia]] for dia in dias_orden}
})

st.dataframe(
    tabla_prod_efectiva.style.format({dia: "{:.2f}" for dia in dias_orden}),
    use_container_width=True
)

# --------------------------------------------------
# 2. HORAS TEÓRICAS (BASADAS EN DISTRIBUCIÓN)
# --------------------------------------------------
st.markdown("---")
st.header("⏰ Horas Teóricas")

# Crear tablas con horas reales distribuidas
tabla_horas_sala_real = pd.DataFrame(
    [horas_sala_por_dia.values],
    index=[f"HORAS {tipo_productividad}"],
    columns=dias_orden
)

tabla_horas_cocina_real = pd.DataFrame(
    [horas_cocina_por_dia.values],
    index=[f"HORAS {tipo_productividad}"],
    columns=dias_orden
)

col3, col4 = st.columns(2)

with col3:
    st.subheader("🍽️ SALA")
    st.dataframe(tabla_horas_sala_real.round(2), use_container_width=True)
    st.caption(f"Total semanal: {horas_sala_por_dia.sum():.2f} horas")

with col4:
    st.subheader("👨‍🍳 COCINA")
    st.dataframe(tabla_horas_cocina_real.round(2), use_container_width=True)
    st.caption(f"Total semanal: {horas_cocina_por_dia.sum():.2f} horas")

# --------------------------------------------------
# 3. DISTRIBUCIÓN DE HORAS POR BLOQUES (TEÓRICA)
# --------------------------------------------------
st.markdown("---")
st.header("📊 Distribución Teórica de Horas por Bloques de 30 min")

area_seleccionada = st.selectbox(
    "Selecciona área para visualizar",
    ["SALA", "COCINA"]
)

if area_seleccionada == "SALA":
    matriz_horas_area = matriz_horas_sala
    horas_semanales_area = horas_semanales_sala
    df_area = df_sala
else:
    matriz_horas_area = matriz_horas_cocina
    horas_semanales_area = horas_semanales_cocina
    df_area = df_cocina

st.info(f"**Horas semanales totales ({area_seleccionada}):** {horas_semanales_area:.2f} horas")

# 3.1. Mapa de calor (días en X, horas en Y)
st.subheader("🕐 Distribución de Horas Requeridas")

suma_matriz = matriz_horas_area.sum().sum()
st.caption(f"✅ Verificación: Suma de horas distribuidas = {suma_matriz:.2f} horas")

fig = px.imshow(
    matriz_horas_area,
    labels=dict(x="Día de la semana", y="Hora del día", color="Horas"),
    x=matriz_horas_area.columns,
    y=matriz_horas_area.index,
    color_continuous_scale="YlOrRd",
    aspect="auto",
    title=f"Distribución de Horas - {area_seleccionada} ({local})"
)

fig.update_layout(
    height=800,
    xaxis_title="Día de la semana",
    yaxis_title="Hora del día",
    yaxis=dict(autorange="reversed"),
    xaxis=dict(side="bottom")
)

fig.update_traces(
    text=matriz_horas_area.round(1),
    texttemplate="%{text}",
    textfont={"size": 7},
    hovertemplate='Día: %{x}<br>Hora: %{y}<br>Horas: %{z:.2f}<extra></extra>'
)

st.plotly_chart(fig, use_container_width=True)

# 3.2. Evolución de demanda (filtrado desde 10:00)
st.subheader("📈 Evolución de demanda durante el día")

df_area_filtrado = df_area.copy()
df_area_filtrado['hora_num'] = pd.to_datetime(df_area_filtrado['bloque_30min'], format='%H:%M').dt.hour
df_area_filtrado = df_area_filtrado[df_area_filtrado['hora_num'] >= 10].copy()

fig_lineas = px.line(
    df_area_filtrado,
    x='bloque_30min',
    y='horas_bloque',
    color='día',
    title=f"Evolución horaria de demanda desde 10:00 - {area_seleccionada}",
    labels={'bloque_30min': 'Hora', 'horas_bloque': 'Horas requeridas', 'día': 'Día'},
    category_orders={"día": dias_orden}
)

fig_lineas.update_layout(
    height=400,
    hovermode='x unified',
    xaxis=dict(tickangle=45, nticks=25)
)

st.plotly_chart(fig_lineas, use_container_width=True)

# 3.3. Comparativa de horas por día
st.subheader("📊 Comparativa de horas por día")

horas_por_dia_area = matriz_horas_area.sum(axis=0)

fig_barras = px.bar(
    x=dias_orden,
    y=horas_por_dia_area.values,
    title=f"Horas requeridas por día - {area_seleccionada}",
    labels={'x': 'Día', 'y': 'Horas requeridas'},
    color=horas_por_dia_area.values,
    color_continuous_scale="Blues",
    text=horas_por_dia_area.values
)

fig_barras.update_traces(texttemplate='%{text:.1f}h', textposition='outside')
fig_barras.update_layout(height=400, showlegend=False)

st.plotly_chart(fig_barras, use_container_width=True)

# --------------------------------------------------
# 4. DISTRIBUCIÓN DE PERSONAL REQUERIDO (NUEVA SECCIÓN)
# --------------------------------------------------
st.markdown("---")
st.header("👥 Distribución de Personal Requerido por Bloques de 30 min")

st.info("""
**Personal Requerido** se calcula redondeando hacia arriba las horas teóricas y aplicando las siguientes restricciones:

**SALA:**
- 1 persona debe llegar 30 min antes de apertura (2 para LLURIA)
- 2 personas deben quedarse 30 min después de cierre

**COCINA:**
- 2 personas deben llegar 1 hora antes de apertura
- 2 personas deben quedarse 30 min después de cierre
""")

area_personal = st.selectbox(
    "Selecciona área para visualizar personal requerido",
    ["SALA", "COCINA"],
    key="area_personal"
)

if area_personal == "SALA":
    matriz_personal_area = matriz_personal_sala
else:
    matriz_personal_area = matriz_personal_cocina

# 4.1. Mapa de calor de personal
st.subheader(f"👤 Distribución de Personal Requerido - {area_personal}")

fig_personal = px.imshow(
    matriz_personal_area,
    labels=dict(x="Día de la semana", y="Hora del día", color="Personal"),
    x=matriz_personal_area.columns,
    y=matriz_personal_area.index,
    color_continuous_scale="Blues",
    aspect="auto",
    title=f"Personal Requerido - {area_personal} ({local})"
)

fig_personal.update_layout(
    height=800,
    xaxis_title="Día de la semana",
    yaxis_title="Hora del día",
    yaxis=dict(autorange="reversed"),
    xaxis=dict(side="bottom")
)

fig_personal.update_traces(
    text=matriz_personal_area,
    texttemplate="%{text}",
    textfont={"size": 8},
    hovertemplate='Día: %{x}<br>Hora: %{y}<br>Personal: %{z}<extra></extra>'
)

st.plotly_chart(fig_personal, use_container_width=True)

# 4.2. Resumen de horas reales por día
st.subheader("⏰ Horas Reales Totales por Día")

if area_personal == "SALA":
    horas_area = horas_reales_sala
else:
    horas_area = horas_reales_cocina

resumen_horas_reales = pd.DataFrame({
    dia: [horas_area[dia], (horas_area[dia] / horas_area.sum() * 100)]
    for dia in dias_orden
}, index=["Horas reales", "% del total semanal"])

resumen_horas_reales["TOTAL SEMANA"] = [
    horas_area.sum(),
    100.0
]

st.dataframe(
    resumen_horas_reales.style.format("{:.2f}"),
    use_container_width=True
)

# --------------------------------------------------
# 5. DISTRIBUCIÓN DE VENTAS
# --------------------------------------------------
st.markdown("---")
st.header("💰 Distribución Promedio de Ventas Semanales")

# 5.1. Mapa de calor
fig_ventas = px.imshow(
    matriz_ventas,
    labels=dict(x="Día de la semana", y="Hora del día", color="Ventas (€)"),
    x=matriz_ventas.columns,
    y=matriz_ventas.index,
    color_continuous_scale="Greens",
    aspect="auto",
    title=f"Distribución Promedio de Ventas Semanales ({local})"
)

fig_ventas.update_layout(
    height=800,
    xaxis_title="Día de la semana",
    yaxis_title="Hora del día",
    yaxis=dict(autorange="reversed"),
    xaxis=dict(side="bottom")
)

fig_ventas.update_traces(
    text=matriz_ventas.round(0),
    texttemplate="€%{text}",
    textfont={"size": 7},
    hovertemplate='Día: %{x}<br>Hora: %{y}<br>Ventas: €%{z:.2f}<extra></extra>'
)

st.plotly_chart(fig_ventas, use_container_width=True)

# 5.2. Comparativa de ventas por día
st.subheader("📊 Comparativa de ventas por día")

fig_barras_ventas = px.bar(
    x=dias_orden,
    y=ventas_por_dia.values,
    title="Ventas promedio por día de la semana",
    labels={'x': 'Día', 'y': 'Ventas (€)'},
    color=ventas_por_dia.values,
    color_continuous_scale="Greens",
    text=ventas_por_dia.values
)

fig_barras_ventas.update_traces(texttemplate='€%{text:.0f}', textposition='outside')
fig_barras_ventas.update_layout(height=400, showlegend=False)

st.plotly_chart(fig_barras_ventas, use_container_width=True)

# --------------------------------------------------
# 6. PRODUCTIVIDAD EFECTIVA DETALLADA (TEÓRICA)
# --------------------------------------------------
st.markdown("---")
st.header("💼 Productividad Efectiva Detallada por Día (Teórica)")

st.info("""
**Productividad Efectiva Teórica** = Ventas del día / (Horas Teóricas Sala + Horas Teóricas Cocina)

Esta métrica muestra cuántos euros se generan por cada hora trabajada total según el modelo teórico.
""")

productividad_df = pd.DataFrame({
    "Día": dias_orden,
    "Ventas (€)": ventas_por_dia.values,
    "Horas Sala": horas_sala_por_dia.values,
    "Horas Cocina": horas_cocina_por_dia.values,
    "Horas Totales": horas_totales_por_dia.values,
    "Productividad Efectiva (€/h)": productividad_efectiva_por_dia.values
})

total_ventas = ventas_por_dia.sum()
total_horas_sala = horas_sala_por_dia.sum()
total_horas_cocina = horas_cocina_por_dia.sum()
total_horas = horas_totales_por_dia.sum()
productividad_efectiva_promedio = total_ventas / total_horas

total_row_suma = pd.DataFrame({
    "Día": ["TOTAL SEMANAL"],
    "Ventas (€)": [total_ventas],
    "Horas Sala": [total_horas_sala],
    "Horas Cocina": [total_horas_cocina],
    "Horas Totales": [total_horas],
    "Productividad Efectiva (€/h)": [productividad_efectiva_promedio]
})

promedio_row = pd.DataFrame({
    "Día": ["PROMEDIO SEMANAL"],
    "Ventas (€)": [total_ventas / 7],
    "Horas Sala": [total_horas_sala / 7],
    "Horas Cocina": [total_horas_cocina / 7],
    "Horas Totales": [total_horas / 7],
    "Productividad Efectiva (€/h)": [productividad_efectiva_promedio]
})

productividad_df = pd.concat([productividad_df, total_row_suma, promedio_row], ignore_index=True)

st.dataframe(
    productividad_df.style.format({
        "Ventas (€)": "€{:.2f}",
        "Horas Sala": "{:.2f}",
        "Horas Cocina": "{:.2f}",
        "Horas Totales": "{:.2f}",
        "Productividad Efectiva (€/h)": "€{:.2f}"
    }),
    use_container_width=True
)

# 6.1. Gráfico de productividad efectiva
st.subheader("📈 Productividad Efectiva Teórica por Día")

productividad_df_dias = productividad_df[
    ~productividad_df['Día'].isin(['TOTAL SEMANAL', 'PROMEDIO SEMANAL'])
]

fig_prod = px.bar(
    productividad_df_dias,
    x="Día",
    y="Productividad Efectiva (€/h)",
    title="Productividad Efectiva Teórica por Día de la Semana",
    color="Productividad Efectiva (€/h)",
    color_continuous_scale="RdYlGn",
    text="Productividad Efectiva (€/h)"
)

fig_prod.update_traces(texttemplate='€%{text:.2f}', textposition='outside')
fig_prod.update_layout(height=400, showlegend=False)

st.plotly_chart(fig_prod, use_container_width=True)

# 6.2. Comparativa horas vs ventas
st.subheader("📊 Comparativa: Horas vs Ventas por Día (Teórica)")

fig_comparativa = go.Figure()

fig_comparativa.add_trace(go.Bar(
    name='Horas Totales',
    x=productividad_df_dias['Día'],
    y=productividad_df_dias['Horas Totales'],
    yaxis='y',
    marker_color='lightblue'
))

fig_comparativa.add_trace(go.Scatter(
    name='Ventas',
    x=productividad_df_dias['Día'],
    y=productividad_df_dias['Ventas (€)'],
    yaxis='y2',
    marker_color='green',
    line=dict(width=3)
))

fig_comparativa.update_layout(
    title='Relación entre Horas Trabajadas y Ventas (Teórica)',
    xaxis=dict(title='Día'),
    yaxis=dict(title='Horas Totales', side='left'),
    yaxis2=dict(title='Ventas (€)', overlaying='y', side='right'),
    height=400,
    hovermode='x unified'
)

st.plotly_chart(fig_comparativa, use_container_width=True)

# --------------------------------------------------
# 7. PRODUCTIVIDAD EFECTIVA REAL (NUEVA SECCIÓN)
# --------------------------------------------------
st.markdown("---")
st.header("💼 Productividad Efectiva Detallada por Día (REAL)")

st.success("""
**Productividad Efectiva REAL** = Ventas del día / (Horas Reales Sala + Horas Reales Cocina)

Esta métrica muestra cuántos euros se generan por cada hora trabajada total considerando el personal requerido real, incluyendo:
- Redondeo hacia arriba de personal necesario
- Personal adicional antes de apertura
- Personal adicional después de cierre
""")

productividad_real_df = pd.DataFrame({
    "Día": dias_orden,
    "Ventas (€)": ventas_por_dia.values,
    "Horas Sala Real": horas_reales_sala.values,
    "Horas Cocina Real": horas_reales_cocina.values,
    "Horas Totales Real": horas_reales_totales.values,
    "Productividad Efectiva Real (€/h)": productividad_efectiva_real.values
})

total_horas_sala_real = horas_reales_sala.sum()
total_horas_cocina_real = horas_reales_cocina.sum()
total_horas_real = horas_reales_totales.sum()
productividad_efectiva_real_promedio = total_ventas / total_horas_real

total_row_real = pd.DataFrame({
    "Día": ["TOTAL SEMANAL"],
    "Ventas (€)": [total_ventas],
    "Horas Sala Real": [total_horas_sala_real],
    "Horas Cocina Real": [total_horas_cocina_real],
    "Horas Totales Real": [total_horas_real],
    "Productividad Efectiva Real (€/h)": [productividad_efectiva_real_promedio]
})

promedio_row_real = pd.DataFrame({
    "Día": ["PROMEDIO SEMANAL"],
    "Ventas (€)": [total_ventas / 7],
    "Horas Sala Real": [total_horas_sala_real / 7],
    "Horas Cocina Real": [total_horas_cocina_real / 7],
    "Horas Totales Real": [total_horas_real / 7],
    "Productividad Efectiva Real (€/h)": [productividad_efectiva_real_promedio]
})

productividad_real_df = pd.concat([productividad_real_df, total_row_real, promedio_row_real], ignore_index=True)

st.dataframe(
    productividad_real_df.style.format({
        "Ventas (€)": "€{:.2f}",
        "Horas Sala Real": "{:.2f}",
        "Horas Cocina Real": "{:.2f}",
        "Horas Totales Real": "{:.2f}",
        "Productividad Efectiva Real (€/h)": "€{:.2f}"
    }),
    use_container_width=True
)

# 7.1. Gráfico de productividad efectiva real
st.subheader("📈 Productividad Efectiva Real por Día")

productividad_real_df_dias = productividad_real_df[
    ~productividad_real_df['Día'].isin(['TOTAL SEMANAL', 'PROMEDIO SEMANAL'])
]

fig_prod_real = px.bar(
    productividad_real_df_dias,
    x="Día",
    y="Productividad Efectiva Real (€/h)",
    title="Productividad Efectiva Real por Día de la Semana",
    color="Productividad Efectiva Real (€/h)",
    color_continuous_scale="RdYlGn",
    text="Productividad Efectiva Real (€/h)"
)

fig_prod_real.update_traces(texttemplate='€%{text:.2f}', textposition='outside')
fig_prod_real.update_layout(height=400, showlegend=False)

st.plotly_chart(fig_prod_real, use_container_width=True)

# 7.2. Comparativa horas reales vs ventas
st.subheader("📊 Comparativa: Horas Reales vs Ventas por Día")

fig_comparativa_real = go.Figure()

fig_comparativa_real.add_trace(go.Bar(
    name='Horas Totales Reales',
    x=productividad_real_df_dias['Día'],
    y=productividad_real_df_dias['Horas Totales Real'],
    yaxis='y',
    marker_color='orange'
))

fig_comparativa_real.add_trace(go.Scatter(
    name='Ventas',
    x=productividad_real_df_dias['Día'],
    y=productividad_real_df_dias['Ventas (€)'],
    yaxis='y2',
    marker_color='green',
    line=dict(width=3)
))

fig_comparativa_real.update_layout(
    title='Relación entre Horas Trabajadas Reales y Ventas',
    xaxis=dict(title='Día'),
    yaxis=dict(title='Horas Totales Reales', side='left'),
    yaxis2=dict(title='Ventas (€)', overlaying='y', side='right'),
    height=400,
    hovermode='x unified'
)

st.plotly_chart(fig_comparativa_real, use_container_width=True)

# 7.3. Comparativa Teórica vs Real
st.subheader("🔄 Comparativa: Productividad Teórica vs Real")

comparativa_prod = pd.DataFrame({
    "Día": dias_orden,
    "Productividad Teórica (€/h)": productividad_efectiva_por_dia.values,
    "Productividad Real (€/h)": productividad_efectiva_real.values,
    "Diferencia (€/h)": productividad_efectiva_por_dia.values - productividad_efectiva_real.values,
    "% Diferencia": ((productividad_efectiva_por_dia.values - productividad_efectiva_real.values) / 
                     productividad_efectiva_por_dia.values * 100)
})

st.dataframe(
    comparativa_prod.style.format({
        "Productividad Teórica (€/h)": "€{:.2f}",
        "Productividad Real (€/h)": "€{:.2f}",
        "Diferencia (€/h)": "€{:.2f}",
        "% Diferencia": "{:.2f}%"
    }),
    use_container_width=True
)

fig_comparativa_prod = go.Figure()

fig_comparativa_prod.add_trace(go.Bar(
    name='Productividad Teórica',
    x=dias_orden,
    y=productividad_efectiva_por_dia.values,
    marker_color='lightblue'
))

fig_comparativa_prod.add_trace(go.Bar(
    name='Productividad Real',
    x=dias_orden,
    y=productividad_efectiva_real.values,
    marker_color='orange'
))

fig_comparativa_prod.update_layout(
    title='Comparativa: Productividad Teórica vs Real',
    xaxis_title='Día',
    yaxis_title='Productividad (€/h)',
    height=400,
    barmode='group',
    hovermode='x unified'
)

st.plotly_chart(fig_comparativa_prod, use_container_width=True)

# --------------------------------------------------
# 8. EXPORTAR DATOS
# --------------------------------------------------
st.markdown("---")
st.subheader("💾 Exportar datos")

col1, col2, col3, col4 = st.columns(4)

with col1:
    csv_personal_sala = matriz_personal_sala.to_csv(index=True)
    st.download_button(
        label="⬇️ Personal SALA",
        data=csv_personal_sala,
        file_name=f"personal_sala_{local}.csv",
        mime="text/csv"
    )

with col2:
    csv_personal_cocina = matriz_personal_cocina.to_csv(index=True)
    st.download_button(
        label="⬇️ Personal COCINA",
        data=csv_personal_cocina,
        file_name=f"personal_cocina_{local}.csv",
        mime="text/csv"
    )

with col3:
    csv_ventas = matriz_ventas.to_csv(index=True)
    st.download_button(
        label="⬇️ Distribución ventas",
        data=csv_ventas,
        file_name=f"distribucion_ventas_{local}.csv",
        mime="text/csv"
    )

with col4:
    csv_prod_real = productividad_real_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Productividad real",
        data=csv_prod_real,
        file_name=f"productividad_real_{local}.csv",
        mime="text/csv"
    )
