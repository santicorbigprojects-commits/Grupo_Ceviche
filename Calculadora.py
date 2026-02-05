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
st.header("Ventas diarias")
st.dataframe(ventas_df.round(2), use_container_width=True)

st.header("Productividad teórica – SALA")
st.dataframe(tabla_sala.round(2), use_container_width=True)

st.header("Productividad teórica – COCINA")
st.dataframe(tabla_cocina.round(2), use_container_width=True)

st.header("Horas Teóricas – SALA")
st.dataframe(tabla_horas_sala.round(2), use_container_width=True)

st.header("Horas Teóricas – COCINA")
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
    
    # Convertir porcentaje_ventas a numérico
    df['porcentaje_ventas'] = pd.to_numeric(df['porcentaje_ventas'], errors='coerce').fillna(0)
    
    # NORMALIZAR NOMBRES DE DÍAS (agregar tildes)
    df['dia'] = df['dia'].str.strip().str.upper()
    df['dia'] = df['dia'].replace({
        'MIERCOLES': 'MIÉRCOLES',
        'SABADO': 'SÁBADO'
    })
    
    return df

try:
    df_distribucion = cargar_distribucion_ventas()
    
    # Filtrar por local
    df_local = df_distribucion[df_distribucion['local'] == local].copy()
    
    if len(df_local) > 0:
        # Verificar suma
        suma_total = df_local['porcentaje_ventas'].sum()
        st.caption(f"ℹ️ Suma de distribución semanal: {suma_total:.4f} (debe ser ≈ 1.0)")
        
        # Selector de área
        area_seleccionada = st.selectbox(
            "Selecciona área para visualizar",
            ["SALA", "COCINA"]
        )
        
        # Obtener horas semanales
        if area_seleccionada == "SALA":
            horas_semanales = tabla_horas_sala.sum(axis=1).values[0]
        else:
            horas_semanales = tabla_horas_cocina.sum(axis=1).values[0]
        
        st.info(f"**Horas semanales totales ({area_seleccionada}):** {horas_semanales:.2f} horas")
        
        # Calcular horas por bloque
        df_local['horas_bloque'] = df_local['porcentaje_ventas'] * horas_semanales
        
        # Crear matriz pivote
        matriz_horas = df_local.pivot_table(
            index='bloque_30min',
            columns='dia',
            values='horas_bloque',
            fill_value=0
        )
        
        # Ordenar días
        dias_orden = ["LUNES", "MARTES", "MIÉRCOLES", "JUEVES", "VIERNES", "SÁBADO", "DOMINGO"]
        matriz_horas = matriz_horas.reindex(columns=dias_orden, fill_value=0)
        
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
                nticks=20
            )
        )
        
        # Añadir valores en celdas
        fig.update_traces(
            text=matriz_horas.T.round(1),
            texttemplate="%{text}",
            textfont={"size": 8},
            hovertemplate='Día: %{y}<br>Hora: %{x}<br>Horas: %{z:.2f}<extra></extra>'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
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
        
        # Top 10 bloques
        st.subheader("🔥 Top 10 Bloques de Mayor Demanda")
        
        matriz_long = matriz_horas.stack().reset_index()
        matriz_long.columns = ['Bloque', 'Día', 'Horas']
        matriz_long = matriz_long.sort_values('Horas', ascending=False).head(10)
        
        st.dataframe(
            matriz_long.reset_index(drop=True).style.format({"Horas": "{:.2f}"}),
            use_container_width=True
        )
        
        # Gráfico de líneas
        st.subheader("📈 Evolución de demanda durante el día")
        
        fig_lineas = px.line(
            df_local,
            x='bloque_30min',
            y='horas_bloque',
            color='dia',
            title=f"Evolución horaria de demanda - {area_seleccionada}",
            labels={'bloque_30min': 'Hora', 'horas_bloque': 'Horas requeridas', 'dia': 'Día'},
            category_orders={"dia": dias_orden}
        )
        
        fig_lineas.update_layout(
            height=400,
            hovermode='x unified',
            xaxis=dict(tickangle=45, nticks=20)
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
        
        df_local['hora_num'] = pd.to_datetime(
            df_local['bloque_30min'], 
            format='%H:%M'
        ).dt.hour + pd.to_datetime(
            df_local['bloque_30min'], 
            format='%H:%M'
        ).dt.minute / 60
        
        def asignar_franja(hora):
            if 6 <= hora < 12:
                return "Mañana (06:00-12:00)"
            elif 12 <= hora < 16:
                return "Mediodía (12:00-16:00)"
            elif 16 <= hora < 20:
                return "Tarde (16:00-20:00)"
            elif 20 <= hora < 24:
                return "Noche (20:00-24:00)"
            else:
                return "Madrugada (00:00-06:00)"
        
        df_local['franja'] = df_local['hora_num'].apply(asignar_franja)
        
        franjas_pivot = df_local.pivot_table(
            index='franja',
            columns='dia',
            values='horas_bloque',
            aggfunc='sum',
            fill_value=0
        )
        
        franjas_pivot = franjas_pivot.reindex(columns=dias_orden, fill_value=0)
        franjas_pivot['Total'] = franjas_pivot.sum(axis=1)
        
        orden_franjas = [
            "Mañana (06:00-12:00)",
            "Mediodía (12:00-16:00)", 
            "Tarde (16:00-20:00)",
            "Noche (20:00-24:00)",
            "Madrugada (00:00-06:00)"
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
                label="⬇️ Distribución detallada",
                data=csv_horas,
                file_name=f"distribucion_horas_{local}_{area_seleccionada}.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_resumen = resumen_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Resumen por día",
                data=csv_resumen,
                file_name=f"resumen_horas_{local}_{area_seleccionada}.csv",
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

