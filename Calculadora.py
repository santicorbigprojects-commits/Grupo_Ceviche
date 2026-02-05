import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
# CÁLCULO PRODUCTIVIDAD SALA
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

# --------------------------------------------------
# CÁLCULO PRODUCTIVIDAD COCINA
# --------------------------------------------------
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

# Renombrar índices
tabla_horas_sala.index = [f"HORAS {tipo_productividad}"]
tabla_horas_cocina.index = [f"HORAS {tipo_productividad}"]

# --------------------------------------------------
# OUTPUTS - TABLAS PRINCIPALES
# --------------------------------------------------
st.header("💰 Ventas diarias")
st.dataframe(ventas_df.round(2), use_container_width=True)

st.markdown("---")
st.header("📊 Productividad Teórica")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🍽️ SALA")
    st.dataframe(tabla_sala.round(2), use_container_width=True)

with col2:
    st.subheader("👨‍🍳 COCINA")
    st.dataframe(tabla_cocina.round(2), use_container_width=True)

st.markdown("---")
st.header("⏰ Horas Teóricas")

col3, col4 = st.columns(2)

with col3:
    st.subheader("🍽️ SALA")
    st.dataframe(tabla_horas_sala.round(2), use_container_width=True)

with col4:
    st.subheader("👨‍🍳 COCINA")
    st.dataframe(tabla_horas_cocina.round(2), use_container_width=True)

# --------------------------------------------------
# MAPA DE CALOR - DISTRIBUCIÓN DE HORAS
# --------------------------------------------------
st.markdown("---")
st.header("📊 Distribución Teórica de Horas por Bloques de 30 min")

# Cargar distribución de ventas
@st.cache_data
def cargar_distribucion_ventas():
    # Intentar leer con diferentes separadores
    try:
        df = pd.read_csv('data/distribucion_ventas_local.csv', sep=';')
    except:
        try:
            df = pd.read_csv('data/distribucion_ventas_local.csv', sep='\t')
        except:
            df = pd.read_csv('data/distribucion_ventas_local.csv', sep=',')
    
    # Convertir bloque_30min a formato HH:MM
    df['bloque_30min'] = pd.to_datetime(df['bloque_30min'], format='%H:%M:%S').dt.strftime('%H:%M')
    
    # Limpiar columnas
    df.columns = df.columns.str.strip()
    
    # Normalizar nombre de columna día/dia
    if 'dia' in df.columns:
        df = df.rename(columns={'dia': 'día'})
    
    # Convertir porcentaje_ventas a numérico
    df['porcentaje_ventas'] = pd.to_numeric(df['porcentaje_ventas'], errors='coerce').fillna(0)
    
    # NORMALIZAR NOMBRES DE DÍAS (agregar tildes)
    df['día'] = df['día'].str.strip().str.upper()
    df['día'] = df['día'].replace({
        'MIERCOLES': 'MIÉRCOLES',
        'SABADO': 'SÁBADO'
    })
    
    # Normalizar columna Distribución (si existe)
    if 'Distribución' in df.columns:
        df['Distribución'] = df['Distribución'].str.strip()
    elif 'Distribucion' in df.columns:
        df = df.rename(columns={'Distribucion': 'Distribución'})
        df['Distribución'] = df['Distribución'].str.strip()
    
    return df

def ordenar_bloques_horarios(bloques):
    """
    Ordena los bloques poniendo los de 8:00-23:30 primero,
    luego 0:00-1:30 al final
    """
    # Convertir a datetime para ordenar
    bloques_dt = pd.to_datetime(bloques, format='%H:%M')
    
    # Separar bloques
    bloques_dia = []  # 8:00 - 23:30
    bloques_noche = []  # 0:00 - 1:30
    
    for i, bloque in enumerate(bloques):
        hora = bloques_dt[i].hour
        if 2 <= hora < 8:  # Filtrar 2:00-7:30 (no hay ventas)
            continue
        elif hora >= 8 or hora == 0:  # 8:00-23:30 o 0:00
            if hora >= 8:
                bloques_dia.append(bloque)
            else:  # 0:00-1:30
                bloques_noche.append(bloque)
        elif hora == 1:  # 1:00-1:30
            bloques_noche.append(bloque)
    
    # Ordenar cada grupo
    bloques_dia_sorted = sorted(bloques_dia, key=lambda x: pd.to_datetime(x, format='%H:%M'))
    bloques_noche_sorted = sorted(bloques_noche, key=lambda x: pd.to_datetime(x, format='%H:%M'))
    
    return bloques_dia_sorted + bloques_noche_sorted

try:
    df_distribucion = cargar_distribucion_ventas()
    
    # Filtrar por local
    df_local = df_distribucion[df_distribucion['local'] == local].copy()
    
    if len(df_local) > 0:
        # Selector de área
        area_seleccionada = st.selectbox(
            "Selecciona área para visualizar",
            ["SALA", "COCINA"]
        )
        
        # Seleccionar tipo de distribución según área
        if area_seleccionada == "SALA":
            tipo_distribucion = "local"
            horas_semanales = tabla_horas_sala.sum(axis=1).values[0]
        else:  # COCINA
            tipo_distribucion = "glovo&local"
            horas_semanales = tabla_horas_cocina.sum(axis=1).values[0]
        
        # Filtrar por tipo de distribución
        df_filtrado = df_local[df_local['Distribución'] == tipo_distribucion].copy()
        
        # Verificar suma
        suma_total = df_filtrado['porcentaje_ventas'].sum()
        st.caption(f"ℹ️ Suma de distribución semanal ({tipo_distribucion}): {suma_total:.4f} (debe ser ≈ 1.0)")
        
        st.info(f"**Horas semanales totales ({area_seleccionada}):** {horas_semanales:.2f} horas")
        
        # ==========================================
        # MAPA DE CALOR 1: DISTRIBUCIÓN DE HORAS
        # ==========================================
        st.subheader("🕐 Distribución de Horas Requeridas")
        
        # Calcular horas por bloque
        df_filtrado['horas_bloque'] = df_filtrado['porcentaje_ventas'] * horas_semanales
        
        # Filtrar bloques entre 2:00-7:59 (excluir horas sin ventas)
        df_filtrado['hora_num'] = pd.to_datetime(df_filtrado['bloque_30min'], format='%H:%M').dt.hour
        df_filtrado = df_filtrado[~((df_filtrado['hora_num'] >= 2) & (df_filtrado['hora_num'] < 8))].copy()
        
        # Crear matriz pivote
        matriz_horas = df_filtrado.pivot_table(
            index='bloque_30min',
            columns='día',
            values='horas_bloque',
            fill_value=0
        )
        
        # Ordenar días
        dias_orden = ["LUNES", "MARTES", "MIÉRCOLES", "JUEVES", "VIERNES", "SÁBADO", "DOMINGO"]
        matriz_horas = matriz_horas.reindex(columns=dias_orden, fill_value=0)
        
        # Reordenar bloques (8:00-23:30, luego 0:00-1:30)
        bloques_ordenados = ordenar_bloques_horarios(matriz_horas.index.tolist())
        matriz_horas = matriz_horas.reindex(bloques_ordenados)
        
        # Verificación
        suma_matriz = matriz_horas.sum().sum()
        st.caption(f"✅ Verificación: Suma de horas distribuidas = {suma_matriz:.2f} horas")
        
        # Crear mapa de calor
        fig = px.imshow(
            matriz_horas.T,
            labels=dict(x="Bloque de 30 min", y="Día", color="Horas"),
            x=matriz_horas.index,
            y=matriz_horas.columns,
            color_continuous_scale="YlOrRd",
            aspect="auto",
            title=f"Distribución de Horas - {area_seleccionada} ({local})"
        )
        
        # Configurar ejes
        fig.update_layout(
            height=500,
            xaxis_title="Hora del día",
            yaxis_title="Día de la semana",
            xaxis=dict(
                side="bottom",
                tickangle=45,
                tickmode='auto',
                nticks=25
            )
        )
        
        # Añadir valores en celdas
        fig.update_traces(
            text=matriz_horas.T.round(1),
            texttemplate="%{text}",
            textfont={"size": 7},
            hovertemplate='Día: %{y}<br>Hora: %{x}<br>Horas: %{z:.2f}<extra></extra>'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ==========================================
        # MAPA DE CALOR 2: DISTRIBUCIÓN DE VENTAS
        # ==========================================
        st.markdown("---")
        st.subheader("💰 Distribución Promedio de Ventas Semanales")
        
        # Filtrar distribución glovo&local para ventas
        df_ventas = df_local[df_local['Distribución'] == "glovo&local"].copy()
        
        # Calcular ventas por bloque
        venta_semanal = venta_diaria * 7
        df_ventas['venta_bloque'] = df_ventas['porcentaje_ventas'] * venta_semanal
        
        # Filtrar bloques entre 2:00-7:59
        df_ventas['hora_num'] = pd.to_datetime(df_ventas['bloque_30min'], format='%H:%M').dt.hour
        df_ventas = df_ventas[~((df_ventas['hora_num'] >= 2) & (df_ventas['hora_num'] < 8))].copy()
        
        # Crear matriz pivote
        matriz_ventas = df_ventas.pivot_table(
            index='bloque_30min',
            columns='día',
            values='venta_bloque',
            fill_value=0
        )
        
        # Ordenar días
        matriz_ventas = matriz_ventas.reindex(columns=dias_orden, fill_value=0)
        
        # Reordenar bloques
        bloques_ordenados_ventas = ordenar_bloques_horarios(matriz_ventas.index.tolist())
        matriz_ventas = matriz_ventas.reindex(bloques_ordenados_ventas)
        
        # Crear mapa de calor de ventas
        fig_ventas = px.imshow(
            matriz_ventas.T,
            labels=dict(x="Bloque de 30 min", y="Día", color="Ventas (€)"),
            x=matriz_ventas.index,
            y=matriz_ventas.columns,
            color_continuous_scale="Greens",
            aspect="auto",
            title=f"Distribución Promedio de Ventas Semanales ({local})"
        )
        
        # Configurar ejes
        fig_ventas.update_layout(
            height=500,
            xaxis_title="Hora del día",
            yaxis_title="Día de la semana",
            xaxis=dict(
                side="bottom",
                tickangle=45,
                tickmode='auto',
                nticks=25
            )
        )
        
        # Añadir valores en celdas
        fig_ventas.update_traces(
            text=matriz_ventas.T.round(0),
            texttemplate="€%{text}",
            textfont={"size": 7},
            hovertemplate='Día: %{y}<br>Hora: %{x}<br>Ventas: €%{z:.2f}<extra></extra>'
        )
        
        st.plotly_chart(fig_ventas, use_container_width=True)
        
        # ==========================================
        # RESÚMENES Y ANÁLISIS
        # ==========================================
        
        # Resumen por día
        st.subheader("📋 Resumen de horas por día")
        
        horas_por_dia = matriz_horas.sum(axis=0)
        
        resumen_df = pd.DataFrame({
            "Día": dias_orden,
            "Horas requeridas": horas_por_dia.values,
            "% del total semanal": (horas_por_dia.values / horas_semanales * 100)
        })
        
        total_row = pd.DataFrame({
            "Día": ["TOTAL SEMANA"],
            "Horas requeridas": [horas_por_dia.sum()],
            "% del total semanal": [100.0]
        })
        resumen_df = pd.concat([resumen_df, total_row], ignore_index=True)
        
        st.dataframe(
            resumen_df.style.format({
                "Horas requeridas": "{:.2f}",
                "% del total semanal": "{:.1f}%"
            }),
            use_container_width=True
        )
        
        # Gráfico de líneas
        st.subheader("📈 Evolución de demanda durante el día")
        
        fig_lineas = px.line(
            df_filtrado,
            x='bloque_30min',
            y='horas_bloque',
            color='día',
            title=f"Evolución horaria de demanda - {area_seleccionada}",
            labels={'bloque_30min': 'Hora', 'horas_bloque': 'Horas requeridas', 'día': 'Día'},
            category_orders={"día": dias_orden}
        )
        
        fig_lineas.update_layout(
            height=400,
            hovermode='x unified',
            xaxis=dict(tickangle=45, nticks=25)
        )
        
        st.plotly_chart(fig_lineas, use_container_width=True)
        
        # Gráfico de barras
        st.subheader("📊 Comparativa de horas por día")
        
        fig_barras = px.bar(
            resumen_df[resumen_df['Día'] != 'TOTAL SEMANA'],
            x="Día",
            y="Horas requeridas",
            title=f"Horas requeridas por día - {area_seleccionada}",
            color="Horas requeridas",
            color_continuous_scale="Blues",
            text="Horas requeridas"
        )
        
        fig_barras.update_traces(texttemplate='%{text:.1f}h', textposition='outside')
        fig_barras.update_layout(height=400)
        
        st.plotly_chart(fig_barras, use_container_width=True)
        
        # Análisis por franjas
        st.subheader("⏰ Análisis por franjas horarias")
        
        df_filtrado['hora_num_completa'] = pd.to_datetime(
            df_filtrado['bloque_30min'], 
            format='%H:%M'
        ).dt.hour + pd.to_datetime(
            df_filtrado['bloque_30min'], 
            format='%H:%M'
        ).dt.minute / 60
        
        def asignar_franja(hora):
            if 8 <= hora < 12:
                return "Mañana (08:00-12:00)"
            elif 12 <= hora < 16:
                return "Mediodía (12:00-16:00)"
            elif 16 <= hora < 20:
                return "Tarde (16:00-20:00)"
            elif 20 <= hora < 24:
                return "Noche (20:00-24:00)"
            else:
                return "Madrugada (00:00-02:00)"
        
        df_filtrado['franja'] = df_filtrado['hora_num_completa'].apply(asignar_franja)
        
        franjas_pivot = df_filtrado.pivot_table(
            index='franja',
            columns='día',
            values='horas_bloque',
            aggfunc='sum',
            fill_value=0
        )
        
        franjas_pivot = franjas_pivot.reindex(columns=dias_orden, fill_value=0)
        franjas_pivot['Total'] = franjas_pivot.sum(axis=1)
        
        orden_franjas = [
            "Mañana (08:00-12:00)",
            "Mediodía (12:00-16:00)", 
            "Tarde (16:00-20:00)",
            "Noche (20:00-24:00)",
            "Madrugada (00:00-02:00)"
        ]
        
        franjas_pivot = franjas_pivot.reindex(
            [f for f in orden_franjas if f in franjas_pivot.index]
        )
        
        st.dataframe(franjas_pivot.round(2), use_container_width=True)
        
        # Exportar
        st.markdown("---")
        st.subheader("💾 Exportar datos")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_horas = matriz_horas.to_csv(index=True)
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
            csv_franjas = franjas_pivot.to_csv(index=True)
            st.download_button(
                label="⬇️ Análisis por franjas",
                data=csv_franjas,
                file_name=f"franjas_horas_{local}_{area_seleccionada}.csv",
                mime="text/csv"
            )
    
    else:
        st.warning(f"⚠️ No hay datos de distribución disponibles para {local}")

except FileNotFoundError:
    st.error("⚠️ Archivo no encontrado. Por favor, asegúrate de que el archivo data/distribucion_ventas_local.csv existe en tu repositorio.")
except Exception as e:
    st.error(f"❌ Error al cargar datos: {str(e)}")
# ==========================================
# PRODUCTIVIDAD EFECTIVA POR DÍA
# ==========================================
st.markdown("---")
st.header("💼 Productividad Efectiva por Día")
        
st.info("""
**Productividad Efectiva** = Ventas del día / (Horas Sala + Horas Cocina)
        
Esta métrica muestra cuántos euros se generan por cada hora trabajada total (sala + cocina).
""")
        
# Calcular acumulados por día para SALA
horas_sala_por_dia = matriz_horas.sum(axis=0) if area_seleccionada == "SALA" else None
        
# Necesitamos calcular también para el otro área
# Obtener datos de la otra área
if area_seleccionada == "SALA":
    # Necesitamos datos de COCINA
    tipo_distribucion_cocina = "glovo&local"
    horas_semanales_cocina = tabla_horas_cocina.sum(axis=1).values[0]
            
    df_cocina = df_local[df_local['Distribución'] == tipo_distribucion_cocina].copy()
    df_cocina['horas_bloque'] = df_cocina['porcentaje_ventas'] * horas_semanales_cocina
    df_cocina['hora_num'] = pd.to_datetime(df_cocina['bloque_30min'], format='%H:%M').dt.hour
    df_cocina = df_cocina[~((df_cocina['hora_num'] >= 2) & (df_cocina['hora_num'] < 8))].copy()
            
    matriz_horas_cocina = df_cocina.pivot_table(
        index='bloque_30min',
        columns='día',
        values='horas_bloque',
        fill_value=0
    )
    matriz_horas_cocina = matriz_horas_cocina.reindex(columns=dias_orden, fill_value=0)
            
    horas_sala_por_dia = matriz_horas.sum(axis=0)
    horas_cocina_por_dia = matriz_horas_cocina.sum(axis=0)
else:
     # Necesitamos datos de SALA
    tipo_distribucion_sala = "local"
    horas_semanales_sala = tabla_horas_sala.sum(axis=1).values[0]
            
    df_sala = df_local[df_local['Distribución'] == tipo_distribucion_sala].copy()
    df_sala['horas_bloque'] = df_sala['porcentaje_ventas'] * horas_semanales_sala
    df_sala['hora_num'] = pd.to_datetime(df_sala['bloque_30min'], format='%H:%M').dt.hour
    df_sala = df_sala[~((df_sala['hora_num'] >= 2) & (df_sala['hora_num'] < 8))].copy()
            
    matriz_horas_sala = df_sala.pivot_table(
        index='bloque_30min',
        columns='día',
        values='horas_bloque',
        fill_value=0
    )
    matriz_horas_sala = matriz_horas_sala.reindex(columns=dias_orden, fill_value=0)
            
    horas_sala_por_dia = matriz_horas_sala.sum(axis=0)
    horas_cocina_por_dia = matriz_horas.sum(axis=0)
        
# Calcular ventas por día desde matriz_ventas
ventas_por_dia = matriz_ventas.sum(axis=0)
        
# Calcular productividad efectiva
horas_totales_por_dia = horas_sala_por_dia + horas_cocina_por_dia
productividad_efectiva = ventas_por_dia / horas_totales_por_dia
        
# Crear DataFrame resumen
productividad_df = pd.DataFrame({
    "Día": dias_orden,
    "Ventas (€)": ventas_por_dia.values,
    "Horas Sala": horas_sala_por_dia.values,
    "Horas Cocina": horas_cocina_por_dia.values,
    "Horas Totales": horas_totales_por_dia.values,
    "Productividad Efectiva (€/h)": productividad_efectiva.values
})
        
# Añadir fila de TOTAL/PROMEDIO
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
        
# Mostrar tabla con estilo
st.dataframe(
    productividad_df.style.format({
        "Ventas (€)": "€{:.2f}",
        "Horas Sala": "{:.2f}",
        "Horas Cocina": "{:.2f}",
        "Horas Totales": "{:.2f}",
        "Productividad Efectiva (€/h)": "€{:.2f}"
    }).background_gradient(subset=["Productividad Efectiva (€/h)"], cmap="RdYlGn"),
    use_container_width=True
)
        
# Gráfico de productividad efectiva
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
fig_prod.update_layout(height=400)
        
st.plotly_chart(fig_prod, use_container_width=True)
        
# Comparativa visual
st.subheader("📊 Comparativa: Horas vs Ventas por Día")
        
fig_comparativa = go.Figure()
        
# Barras de horas totales
fig_comparativa.add_trace(go.Bar(
    name='Horas Totales',
    x=productividad_df[productividad_df['Día'] != 'PROMEDIO SEMANAL']['Día'],
    y=productividad_df[productividad_df['Día'] != 'PROMEDIO SEMANAL']['Horas Totales'],
    yaxis='y',
    marker_color='lightblue'
))
        
# Línea de ventas
fig_comparativa.add_trace(go.Scatter(
    name='Ventas',
    x=productividad_df[productividad_df['Día'] != 'PROMEDIO SEMANAL']['Día'],
    y=productividad_df[productividad_df['Día'] != 'PROMEDIO SEMANAL']['Ventas (€)'],
    yaxis='y2',
    marker_color='green',
    line=dict(width=3)
))
        
# Configurar ejes
fig_comparativa.update_layout(
    title='Relación entre Horas Trabajadas y Ventas',
    xaxis=dict(title='Día'),
    yaxis=dict(
        title='Horas Totales',
        side='left'
    ),
    yaxis2=dict(
        title='Ventas (€)',
        overlaying='y',
        side='right'
    ),
    height=400,
    hovermode='x unified'
)
        
st.plotly_chart(fig_comparativa, use_container_width=True)
