import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
# OUTPUTS - TABLAS PRINCIPALES
# --------------------------------------------------
st.header("💰 Ventas diarias")
st.dataframe(ventas_df.round(2), use_container_width=True)

# --------------------------------------------------
# 1. PRODUCTIVIDAD EFECTIVA
# --------------------------------------------------
st.markdown("---")
st.header("💼 Productividad Efectiva")
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
# 2. HORAS TEÓRICAS (AHORA BASADAS EN DISTRIBUCIÓN REAL)
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
# 3. DISTRIBUCIÓN DE HORAS POR BLOQUES
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

# 3.1. Resumen de horas por día (transpuesto)
st.subheader("📋 Resumen de horas por día")

horas_por_dia_area = matriz_horas_area.sum(axis=0)

resumen_transpuesto = pd.DataFrame({
    dia: [horas_por_dia_area[dia], (horas_por_dia_area[dia] / horas_semanales_area * 100)] 
    for dia in dias_orden
}, index=["Horas requeridas", "% del total semanal"])

# Agregar columna TOTAL
resumen_transpuesto["TOTAL SEMANA"] = [
    horas_por_dia_area.sum(),
    100.0
]

st.dataframe(
    resumen_transpuesto.style.format("{:.2f}"),
    use_container_width=True
)

# 3.2. Mapa de calor (días en X, horas en Y)
st.subheader("🕐 Distribución de Horas Requeridas")

suma_matriz = matriz_horas_area.sum().sum()
st.caption(f"✅ Verificación: Suma de horas distribuidas = {suma_matriz:.2f} horas")

fig = px.imshow(
    matriz_horas_area,  # SIN transponer (bloques en Y, días en X)
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
    yaxis=dict(autorange="reversed"),  # Invertir Y para que 8:00 esté arriba
    xaxis=dict(side="bottom")
)

fig.update_traces(
    text=matriz_horas_area.round(1),
    texttemplate="%{text}",
    textfont={"size": 7},
    hovertemplate='Día: %{x}<br>Hora: %{y}<br>Horas: %{z:.2f}<extra></extra>'
)

st.plotly_chart(fig, use_container_width=True)

# 3.3. Evolución de demanda (filtrado desde 10:00)
st.subheader("📈 Evolución de demanda durante el día")

# Filtrar desde las 10:00
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

# 3.4. Comparativa de horas por día
st.subheader("📊 Comparativa de horas por día")

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
# 4. DISTRIBUCIÓN DE VENTAS
# --------------------------------------------------
st.markdown("---")
st.header("💰 Distribución Promedio de Ventas Semanales")

# 4.1. Mapa de calor (días en X, horas en Y)
fig_ventas = px.imshow(
    matriz_ventas,  # SIN transponer
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
    yaxis=dict(autorange="reversed"),  # Invertir Y para que 8:00 esté arriba
    xaxis=dict(side="bottom")
)

fig_ventas.update_traces(
    text=matriz_ventas.round(0),
    texttemplate="€%{text}",
    textfont={"size": 7},
    hovertemplate='Día: %{x}<br>Hora: %{y}<br>Ventas: €%{z:.2f}<extra></extra>'
)

st.plotly_chart(fig_ventas, use_container_width=True)

# 4.2. Comparativa de ventas por día
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
# 5. PRODUCTIVIDAD EFECTIVA DETALLADA
# --------------------------------------------------
st.markdown("---")
st.header("💼 Productividad Efectiva Detallada por Día")

st.info("""
**Productividad Efectiva** = Ventas del día / (Horas Sala + Horas Cocina)

Esta métrica muestra cuántos euros se generan por cada hora trabajada total.
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

total_row_prod = pd.DataFrame({
    "Día": ["PROMEDIO SEMANAL"],
    "Ventas (€)": [total_ventas / 7],
    "Horas Sala": [total_horas_sala / 7],
    "Horas Cocina": [total_horas_cocina / 7],
    "Horas Totales": [total_horas / 7],
    "Productividad Efectiva (€/h)": [productividad_efectiva_promedio]
})

productividad_df = pd.concat([productividad_df, total_row_prod], ignore_index=True)

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

# 5.1. Gráfico de productividad efectiva
st.subheader("📈 Productividad Efectiva por Día")

fig_prod = px.bar(
    productividad_df[productividad_df['Día'] != 'PROMEDIO SEMANAL'],
    x="Día",
    y="Productividad Efectiva (€/h)",
    title="Productividad Efectiva por Día de la Semana",
    color="Productividad Efectiva (€/h)",
    color_continuous_scale="RdYlGn",
    text="Productividad Efectiva (€/h)"
)

fig_prod.update_traces(texttemplate='€%{text:.2f}', textposition='outside')
fig_prod.update_layout(height=400, showlegend=False)

st.plotly_chart(fig_prod, use_container_width=True)

# 5.2. Comparativa horas vs ventas
st.subheader("📊 Comparativa: Horas vs Ventas por Día")

fig_comparativa = go.Figure()

fig_comparativa.add_trace(go.Bar(
    name='Horas Totales',
    x=productividad_df[productividad_df['Día'] != 'PROMEDIO SEMANAL']['Día'],
    y=productividad_df[productividad_df['Día'] != 'PROMEDIO SEMANAL']['Horas Totales'],
    yaxis='y',
    marker_color='lightblue'
))

fig_comparativa.add_trace(go.Scatter(
    name='Ventas',
    x=productividad_df[productividad_df['Día'] != 'PROMEDIO SEMANAL']['Día'],
    y=productividad_df[productividad_df['Día'] != 'PROMEDIO SEMANAL']['Ventas (€)'],
    yaxis='y2',
    marker_color='green',
    line=dict(width=3)
))

fig_comparativa.update_layout(
    title='Relación entre Horas Trabajadas y Ventas',
    xaxis=dict(title='Día'),
    yaxis=dict(title='Horas Totales', side='left'),
    yaxis2=dict(title='Ventas (€)', overlaying='y', side='right'),
    height=400,
    hovermode='x unified'
)

st.plotly_chart(fig_comparativa, use_container_width=True)

# --------------------------------------------------
# 6. EXPORTAR DATOS
# --------------------------------------------------
st.markdown("---")
st.subheader("💾 Exportar datos")

col1, col2, col3 = st.columns(3)

with col1:
    csv_horas = matriz_horas_area.to_csv(index=True)
    st.download_button(
        label="⬇️ Distribución horas",
        data=csv_horas,
        file_name=f"distribucion_horas_{local}_{area_seleccionada}.csv",
        mime="text/csv"
    )

with col2:
    csv_ventas = matriz_ventas.to_csv(index=True)
    st.download_button(
        label="⬇️ Distribución ventas",
        data=csv_ventas,
        file_name=f"distribucion_ventas_{local}.csv",
        mime="text/csv"
    )

with col3:
    csv_prod = productividad_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Productividad efectiva",
        data=csv_prod,
        file_name=f"productividad_efectiva_{local}.csv",
        mime="text/csv"
    )
# --------------------------------------------------
# 7. PLANIFICADORES SEMANALES POR DÍA
# --------------------------------------------------
st.markdown("---")
st.header("📅 Planificadores de Turnos por Día")

st.info("""
**Asignación automática de turnos** considerando:
- **Turnos continuos:** Máximo 10 bloques (5 horas)
- **Descanso entre turnos:** Mínimo 2 bloques (1 hora)
- **Horas diarias:** Máx. 18 bloques sala (9h) / 20 bloques cocina (10h)
- **Horas semanales:** Máx. 100 bloques sala (50h) / 108 bloques cocina (54h)
- **Descanso semanal:** 2 días completos
""")

# Horarios de apertura y cierre por local
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
        "VIERNES": {"abre": "13:00", "cierra": "0:30"},
        "SÁBADO": {"abre": "13:00", "cierra": "0:30"},
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

# Constantes de restricciones
MAX_BLOQUES_TURNO = 10  # 5 horas
DESCANSO_ENTRE_TURNOS = 2  # 1 hora
MAX_BLOQUES_DIA_SALA = 18  # 9 horas
MAX_BLOQUES_DIA_COCINA = 20  # 10 horas
MAX_BLOQUES_SEMANA_SALA = 100  # 50 horas
MAX_BLOQUES_SEMANA_COCINA = 108  # 54 horas

def calcular_trabajadores_necesarios(horas_requeridas):
    """Redondea hacia arriba las horas requeridas"""
    import math
    return math.ceil(horas_requeridas)

def ajustar_hora_cierre(hora_str):
    """Convierte 0:30 a 24:30 para mantener continuidad"""
    if hora_str.startswith("0:"):
        partes = hora_str.split(":")
        return f"24:{partes[1]}"
    return hora_str

def encontrar_indice_bloque(bloques_lista, hora_objetivo):
    """Encuentra el índice del bloque más cercano a la hora objetivo"""
    from datetime import datetime
    
    if hora_objetivo.startswith("24:"):
        hora_objetivo = hora_objetivo.replace("24:", "0:")
        es_dia_siguiente = True
    else:
        es_dia_siguiente = False
    
    try:
        hora_obj = datetime.strptime(hora_objetivo, "%H:%M")
    except:
        return None
    
    for i, bloque in enumerate(bloques_lista):
        try:
            bloque_obj = datetime.strptime(bloque, "%H:%M")
            
            if es_dia_siguiente and bloque_obj.hour <= 2:
                return i
            elif not es_dia_siguiente and bloque_obj >= hora_obj:
                return i
        except:
            continue
    
    return len(bloques_lista) - 1

def crear_bloques_horario():
    """Crea lista de bloques de 8:00 a 1:30"""
    bloques = []
    
    # 8:00 a 23:30
    for h in range(8, 24):
        for m in [0, 30]:
            bloques.append(f"{h:02d}:{m:02d}")
    
    # 0:00 a 1:30
    for h in [0, 1]:
        for m in [0, 30]:
            bloques.append(f"{h:02d}:{m:02d}")
            if h == 1 and m == 30:
                break
    
    return bloques

def asignar_turnos_optimizado(bloques, demanda_bloques, max_bloques_dia, es_lluria=False, min_apertura=1, min_cierre=2):
    """
    Asigna turnos respetando restricciones de turnos continuos y descansos
    
    Returns: dict con {trabajador: [bloques asignados]}
    """
    num_trabajadores = max(demanda_bloques.values()) if demanda_bloques else 0
    
    # Ajustar por restricciones especiales
    if es_lluria:
        num_trabajadores = max(num_trabajadores, min_apertura)
    else:
        num_trabajadores = max(num_trabajadores, min_cierre)
    
    asignaciones = {i: [] for i in range(num_trabajadores)}
    bloques_cubiertos = {bloque: 0 for bloque in bloques}
    
    # Ordenar trabajadores para distribuir carga
    for trabajador in range(num_trabajadores):
        bloques_consecutivos = 0
        ultimo_bloque_idx = -100
        
        for idx, bloque in enumerate(bloques):
            demanda = demanda_bloques.get(bloque, 0)
            
            # Si ya hay suficientes trabajadores cubriendo este bloque
            if bloques_cubiertos[bloque] >= demanda:
                bloques_consecutivos = 0
                continue
            
            # Si el trabajador ya alcanzó su límite diario
            if len(asignaciones[trabajador]) >= max_bloques_dia:
                break
            
            # Verificar descanso entre turnos (2 bloques)
            if ultimo_bloque_idx >= 0 and idx - ultimo_bloque_idx > DESCANSO_ENTRE_TURNOS + 1:
                bloques_consecutivos = 0
            
            # Si el trabajador puede tomar este bloque
            if bloques_consecutivos < MAX_BLOQUES_TURNO:
                asignaciones[trabajador].append(bloque)
                bloques_cubiertos[bloque] += 1
                bloques_consecutivos += 1
                ultimo_bloque_idx = idx
            else:
                # Necesita descanso
                bloques_consecutivos = 0
    
    return asignaciones

# Crear planificadores por día
bloques_completos = crear_bloques_horario()

# Almacenar asignaciones semanales para verificar límites
asignaciones_semanales_sala = {}
asignaciones_semanales_cocina = {}

for dia in dias_orden:
    st.markdown("---")
    st.subheader(f"📆 {dia}")
    
    if local not in horarios_locales or dia not in horarios_locales[local]:
        st.warning(f"⚠️ No hay horarios configurados para {local} - {dia}")
        continue
    
    hora_apertura = horarios_locales[local][dia]["abre"]
    hora_cierre_raw = horarios_locales[local][dia]["cierra"]
    hora_cierre = ajustar_hora_cierre(hora_cierre_raw)
    
    idx_apertura = encontrar_indice_bloque(bloques_completos, hora_apertura)
    idx_cierre = encontrar_indice_bloque(bloques_completos, hora_cierre)
    
    if idx_apertura is None or idx_cierre is None:
        st.warning(f"⚠️ No se pudo calcular horarios para {dia}")
        continue
    
    # Ajustar índices con llegadas anticipadas
    idx_inicio_sala = max(0, idx_apertura - 1)  # 30 min antes
    idx_inicio_cocina = max(0, idx_apertura - 2)  # 1h antes
    idx_fin_sala = min(len(bloques_completos) - 1, idx_cierre + 1)  # 30 min después
    idx_fin_cocina = min(len(bloques_completos) - 1, idx_cierre + 1)  # 30 min después
    
    bloques_sala = bloques_completos[idx_inicio_sala:idx_fin_sala + 1]
    bloques_cocina = bloques_completos[idx_inicio_cocina:idx_fin_cocina + 1]
    
    # Obtener demanda por bloque
    horas_sala_dia = matriz_horas_sala[dia]
    horas_cocina_dia = matriz_horas_cocina[dia]
    
    demanda_sala = {}
    for bloque in bloques_sala:
        if bloque in horas_sala_dia.index:
            demanda_sala[bloque] = calcular_trabajadores_necesarios(horas_sala_dia[bloque])
        else:
            demanda_sala[bloque] = 0
    
    # Aplicar restricciones de apertura/cierre SALA
    if local == "LLURIA":
        for i in range(min(2, len(bloques_sala))):
            demanda_sala[bloques_sala[i]] = max(demanda_sala.get(bloques_sala[i], 0), 3)
        for i in range(max(0, len(bloques_sala) - 2), len(bloques_sala)):
            demanda_sala[bloques_sala[i]] = max(demanda_sala.get(bloques_sala[i], 0), 4)
    else:
        for i in range(min(2, len(bloques_sala))):
            demanda_sala[bloques_sala[i]] = max(demanda_sala.get(bloques_sala[i], 0), 1)
        for i in range(max(0, len(bloques_sala) - 2), len(bloques_sala)):
            demanda_sala[bloques_sala[i]] = max(demanda_sala.get(bloques_sala[i], 0), 2)
    
    demanda_cocina = {}
    for bloque in bloques_cocina:
        if bloque in horas_cocina_dia.index:
            demanda_cocina[bloque] = calcular_trabajadores_necesarios(horas_cocina_dia[bloque])
        else:
            demanda_cocina[bloque] = 0
    
    # Aplicar restricciones de apertura/cierre COCINA
    if local == "LLURIA":
        for i in range(min(2, len(bloques_cocina))):
            demanda_cocina[bloques_cocina[i]] = max(demanda_cocina.get(bloques_cocina[i], 0), 5)
        for i in range(max(0, len(bloques_cocina) - 2), len(bloques_cocina)):
            demanda_cocina[bloques_cocina[i]] = max(demanda_cocina.get(bloques_cocina[i], 0), 5)
    else:
        for i in range(min(2, len(bloques_cocina))):
            demanda_cocina[bloques_cocina[i]] = max(demanda_cocina.get(bloques_cocina[i], 0), 2)
        for i in range(max(0, len(bloques_cocina) - 2), len(bloques_cocina)):
            demanda_cocina[bloques_cocina[i]] = max(demanda_cocina.get(bloques_cocina[i], 0), 2)
    
    # Asignar turnos SALA
    asignaciones_sala = asignar_turnos_optimizado(
        bloques_sala, 
        demanda_sala, 
        MAX_BLOQUES_DIA_SALA,
        es_lluria=(local == "LLURIA"),
        min_apertura=3 if local == "LLURIA" else 1,
        min_cierre=4 if local == "LLURIA" else 2
    )
    
    # Asignar turnos COCINA
    asignaciones_cocina = asignar_turnos_optimizado(
        bloques_cocina,
        demanda_cocina,
        MAX_BLOQUES_DIA_COCINA,
        es_lluria=(local == "LLURIA"),
        min_apertura=5 if local == "LLURIA" else 2,
        min_cierre=5 if local == "LLURIA" else 2
    )
    
    # Crear DataFrames con trabajadores en filas y bloques en columnas
    st.write(f"**🍽️ SALA** (Apertura: {hora_apertura} | Cierre: {hora_cierre_raw})")
    
    planificador_sala = pd.DataFrame(
        "",
        index=[f"Trabajador Sala {i+1}" for i in range(len(asignaciones_sala))],
        columns=bloques_sala
    )
    
    for trabajador, bloques_asignados in asignaciones_sala.items():
        for bloque in bloques_asignados:
            if bloque in planificador_sala.columns:
                planificador_sala.loc[f"Trabajador Sala {trabajador+1}", bloque] = "✓"
    
    def color_celda(val):
        if val == "✓":
            return 'background-color: #90EE90'
        return ''
    
    st.dataframe(
        planificador_sala.style.applymap(color_celda),
        use_container_width=True,
        height=min(400, len(asignaciones_sala) * 35 + 50)
    )
    
    # Mostrar estadísticas SALA
    bloques_por_trabajador_sala = {f"Trab. {i+1}": len(bloques) for i, bloques in asignaciones_sala.items()}
    st.caption(f"Bloques asignados: {bloques_por_trabajador_sala}")
    
    st.write(f"**👨‍🍳 COCINA** (Apertura: {hora_apertura} | Cierre: {hora_cierre_raw})")
    
    planificador_cocina = pd.DataFrame(
        "",
        index=[f"Trabajador Cocina {i+1}" for i in range(len(asignaciones_cocina))],
        columns=bloques_cocina
    )
    
    for trabajador, bloques_asignados in asignaciones_cocina.items():
        for bloque in bloques_asignados:
            if bloque in planificador_cocina.columns:
                planificador_cocina.loc[f"Trabajador Cocina {trabajador+1}", bloque] = "✓"
    
    st.dataframe(
        planificador_cocina.style.applymap(color_celda),
        use_container_width=True,
        height=min(400, len(asignaciones_cocina) * 35 + 50)
    )
    
    # Mostrar estadísticas COCINA
    bloques_por_trabajador_cocina = {f"Trab. {i+1}": len(bloques) for i, bloques in asignaciones_cocina.items()}
    st.caption(f"Bloques asignados: {bloques_por_trabajador_cocina}")
    
    # Resumen del día
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Trabajadores Sala", len(asignaciones_sala))
    with col2:
        st.metric("Trabajadores Cocina", len(asignaciones_cocina))
    with col3:
        total_bloques_sala = sum(len(b) for b in asignaciones_sala.values())
        st.metric("Total Bloques Sala", f"{total_bloques_sala} ({total_bloques_sala*0.5:.1f}h)")
    with col4:
        total_bloques_cocina = sum(len(b) for b in asignaciones_cocina.values())
        st.metric("Total Bloques Cocina", f"{total_bloques_cocina} ({total_bloques_cocina*0.5:.1f}h)")

st.markdown("---")
st.info("""
**Restricciones aplicadas:**

✅ **Turnos continuos:** Máximo 10 bloques seguidos (5 horas)  
✅ **Descanso entre turnos:** Mínimo 2 bloques (1 hora)  
✅ **Límite diario:** 18 bloques sala / 20 bloques cocina  
✅ **Personal de apertura/cierre:** Según normativa de cada local  
✅ **Llegadas anticipadas:** Sala -1 bloque / Cocina -2 bloques  

⚠️ **Pendiente optimizar:** Límites semanales (100 bloques sala / 108 bloques cocina) y descanso de 2 días
""")
