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
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Calculadora de Eficiencia y Horas – RIKOS")

# --------------------------------------------------
# SIDEBAR - INPUTS
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.subheader("📍 Local")
    local = st.selectbox(
        "Selecciona el local",
        ["MERIDIANA", "CAN VIDALET", "BADAL", "CORNELLA",
         "GLORIES", "SANTA COLOMA", "ICARIA", "LLURIA"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.subheader("💰 Ventas")
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
    
    st.markdown("---")
    st.subheader("📊 Productividad")
    tipo_productividad = st.selectbox(
        "Tipo de productividad",
        ["ESPERADO", "MÁXIMO", "MÍNIMO"]
    )
    
    st.markdown("---")
    st.subheader("👥 Personal de Apertura/Cierre")
    
    with st.expander("🍽️ SALA", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            personal_apertura_sala = st.number_input(
                "Apertura",
                min_value=1,
                value=1,
                step=1,
                key="ap_sala"
            )
        with col2:
            personal_cierre_sala = st.number_input(
                "Cierre",
                min_value=1,
                value=2,
                step=1,
                key="ci_sala"
            )
    
    with st.expander("👨‍🍳 COCINA", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            personal_apertura_cocina = st.number_input(
                "Apertura",
                min_value=1,
                value=2,
                step=1,
                key="ap_cocina"
            )
        with col2:
            personal_cierre_cocina = st.number_input(
                "Cierre",
                min_value=1,
                value=2,
                step=1,
                key="ci_cocina"
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

ventas_df = pd.DataFrame({
    "Concepto": ["Venta total", "Venta sala", "Venta Glovo"],
    "Monto (€)": [venta_diaria, venta_sala, venta_glovo]
})

# --------------------------------------------------
# PARÁMETROS DE LOS MODELOS
# --------------------------------------------------
dias = ["LUNES", "MARTES", "MIÉRCOLES", "JUEVES", "VIERNES", "SÁBADO", "DOMINGO"]

coef_dia_sala = {
    "LUNES": 18.2192, "MARTES": 18.2916, "MIÉRCOLES": 10.8842,
    "JUEVES": 18.4559, "VIERNES": 7.4385, "SÁBADO": 1.5453, "DOMINGO": 0
}

coef_dia_cocina = {
    "LUNES": 6.1412, "MARTES": 4.2809, "MIÉRCOLES": 3.5314,
    "JUEVES": 7.7640, "VIERNES": 1.7950, "SÁBADO": -3.2005, "DOMINGO": 0
}

coef_local_sala = {
    "CAN VIDALET": 47.5779, "CORNELLA": 32.0810, "GLORIES": 30.6683,
    "ICARIA": 26.5032, "LLURIA": -17.7891, "MERIDIANA": 23.1716,
    "SANTA COLOMA": 28.7016, "BADAL": 0
}

coef_local_cocina = {
    "CAN VIDALET": 32.2937, "CORNELLA": 46.5149, "GLORIES": 48.7297,
    "ICARIA": 49.4127, "LLURIA": -17.3252, "MERIDIANA": 10.9482,
    "SANTA COLOMA": 38.5073, "BADAL": 0
}

share_glovo_promedio = {
    "MERIDIANA": 0.203457008, "CAN VIDALET": 0.164354398, "BADAL": 0.442485954,
    "CORNELLA": 0.375005705, "GLORIES": 0.442987607, "SANTA COLOMA": 0.527390956,
    "ICARIA": 0.327043339, "LLURIA": 0.323162578
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
    factor_sala, factor_cocina = ajuste_sala, ajuste_cocina
elif tipo_productividad == "MÍNIMO":
    factor_sala, factor_cocina = -ajuste_sala, -ajuste_cocina
else:
    factor_sala, factor_cocina = 0, 0

# --------------------------------------------------
# CÁLCULO PRODUCTIVIDAD Y HORAS TEÓRICAS
# --------------------------------------------------
prod_sala = []
for d in dias:
    valor = (-426.3496 + coef_dia_sala[d] + 64.4162 * ln_ventas_sala +
             coef_local_sala.get(local, 0) + factor_sala)
    prod_sala.append(valor)

prod_cocina = []
for d in dias:
    valor = (-565.6808 + coef_dia_cocina[d] + 80.0758 * ln_ventas_total +
             16.8018 * share_glovo_centrado + coef_local_cocina.get(local, 0) + factor_cocina)
    prod_cocina.append(valor)

tabla_sala = pd.DataFrame([prod_sala], columns=dias)
tabla_cocina = pd.DataFrame([prod_cocina], columns=dias)

tabla_horas_sala = venta_sala / tabla_sala
tabla_horas_cocina = venta_diaria / tabla_cocina

# --------------------------------------------------
# FUNCIONES AUXILIARES
# --------------------------------------------------
def convertir_hora_a_minutos(hora_str):
    partes = hora_str.split(':')
    return int(partes[0]) * 60 + int(partes[1])

def minutos_a_bloque(minutos):
    horas = minutos // 60
    mins = minutos % 60
    if horas >= 24:
        horas = horas - 24
    return f"{horas}:{mins:02d}"

def ordenar_bloques_horarios(bloques):
    bloques_dt = pd.to_datetime(bloques, format='%H:%M')
    bloques_dia, bloques_noche = [], []
    
    for i, bloque in enumerate(bloques):
        hora = bloques_dt[i].hour
        if 2 <= hora < 8:
            continue
        elif hora >= 8:
            bloques_dia.append(bloque)
        elif hora <= 1:
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
    df['día'] = df['día'].str.strip().str.upper().replace({'MIERCOLES': 'MIÉRCOLES', 'SABADO': 'SÁBADO'})
    
    if 'Distribución' in df.columns:
        df['Distribución'] = df['Distribución'].str.strip()
    elif 'Distribucion' in df.columns:
        df = df.rename(columns={'Distribucion': 'Distribución'})
        df['Distribución'] = df['Distribución'].str.strip()
    
    return df

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
        index='bloque_30min', columns='día', values='horas_bloque', fill_value=0
    )
    
    # Preparar datos COCINA
    df_cocina = df_local[df_local['Distribución'] == "glovo&local"].copy()
    horas_semanales_cocina = tabla_horas_cocina.sum(axis=1).values[0]
    df_cocina['horas_bloque'] = df_cocina['porcentaje_ventas'] * horas_semanales_cocina
    df_cocina['hora_num'] = pd.to_datetime(df_cocina['bloque_30min'], format='%H:%M').dt.hour
    df_cocina = df_cocina[~((df_cocina['hora_num'] >= 2) & (df_cocina['hora_num'] < 8))].copy()
    
    matriz_horas_cocina = df_cocina.pivot_table(
        index='bloque_30min', columns='día', values='horas_bloque', fill_value=0
    )
    
    # Preparar datos VENTAS
    df_ventas = df_local[df_local['Distribución'] == "glovo&local"].copy()
    venta_semanal = venta_diaria * 7
    df_ventas['venta_bloque'] = df_ventas['porcentaje_ventas'] * venta_semanal
    df_ventas['hora_num'] = pd.to_datetime(df_ventas['bloque_30min'], format='%H:%M').dt.hour
    df_ventas = df_ventas[~((df_ventas['hora_num'] >= 2) & (df_ventas['hora_num'] < 8))].copy()
    
    matriz_ventas = df_ventas.pivot_table(
        index='bloque_30min', columns='día', values='venta_bloque', fill_value=0
    )
    
    # Ordenar días y bloques
    dias_orden = ["LUNES", "MARTES", "MIÉRCOLES", "JUEVES", "VIERNES", "SÁBADO", "DOMINGO"]
    matriz_horas_sala = matriz_horas_sala.reindex(columns=dias_orden, fill_value=0)
    matriz_horas_cocina = matriz_horas_cocina.reindex(columns=dias_orden, fill_value=0)
    matriz_ventas = matriz_ventas.reindex(columns=dias_orden, fill_value=0)
    
    bloques_ordenados = ordenar_bloques_horarios(matriz_horas_sala.index.tolist())
    matriz_horas_sala = matriz_horas_sala.reindex(bloques_ordenados)
    matriz_horas_cocina = matriz_horas_cocina.reindex(bloques_ordenados)
    
    bloques_ordenados_ventas = ordenar_bloques_horarios(matriz_ventas.index.tolist())
    matriz_ventas = matriz_ventas.reindex(bloques_ordenados_ventas)

except FileNotFoundError:
    st.error("⚠️ Archivo no encontrado. Asegúrate de que data/distribucion_ventas_local.csv existe.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error al cargar datos: {str(e)}")
    st.stop()

# --------------------------------------------------
# CALCULAR PERSONAL REQUERIDO CON RESTRICCIONES
# --------------------------------------------------
def calcular_personal_requerido(matriz_horas, area, local, dias_orden, pers_apertura, pers_cierre):
    """
    Calcula personal requerido aplicando restricciones de apertura/cierre y máximos
    
    RESTRICCIONES:
    - Apertura: Se aplica en bloques ANTES de abrir
    - Cierre: Se aplica en el bloque de cierre y el anterior (últimos 2 bloques)
    - Máximos de personal por área:
      * Sala: 4 personas (5 en Lluria)
      * Cocina: 5 personas (6 en Lluria)
    - De lunes a viernes: NO se puede alcanzar el máximo (se resta 1)
    - Objetivo: minimizar personal (usar ceil en lugar de redondeos generosos)
    """
    # Convertir horas a personas (ceil para minimizar personal)
    matriz_personal = np.ceil(matriz_horas * 2).astype(int)
    
    # Definir máximos según área y local
    if local == "LLURIA":
        max_personal = 6 if area == "COCINA" else 5
    else:
        max_personal = 5 if area == "COCINA" else 4
    
    # Días laborables donde NO se puede alcanzar el máximo
    dias_laborables = ["LUNES", "MARTES", "MIÉRCOLES", "JUEVES", "VIERNES"]
    
    horarios = horarios_locales[local]
    
    for dia in dias_orden:
        if dia not in horarios:
            continue
        
        # Determinar máximo efectivo según el día
        if dia in dias_laborables:
            max_efectivo = max_personal - 1  # De L-V no se alcanza el máximo
        else:
            max_efectivo = max_personal
        
        # Aplicar límite máximo a toda la columna del día
        matriz_personal[dia] = matriz_personal[dia].clip(upper=max_efectivo)
        
        try:
            hora_abre = horarios[dia]["abre"]
            hora_cierra = horarios[dia]["cierra"]
            
            minutos_abre = convertir_hora_a_minutos(hora_abre)
            minutos_cierra = convertir_hora_a_minutos(hora_cierra)
            
            # APERTURA: 2 bloques ANTES de abrir (1 hora antes)
            for i in range(2):
                minutos_bloque = minutos_abre - (2 - i) * 30
                bloque_str = minutos_a_bloque(minutos_bloque)
                
                if bloque_str in matriz_personal.index:
                    # Aplicar mínimo de apertura, pero respetar máximo efectivo
                    matriz_personal.loc[bloque_str, dia] = min(
                        max(matriz_personal.loc[bloque_str, dia], pers_apertura),
                        max_efectivo
                    )
            
            # CIERRE: En el bloque de cierre Y el anterior (últimos 2 bloques)
            for i in range(2):
                minutos_bloque = minutos_cierra - i * 30
                bloque_str = minutos_a_bloque(minutos_bloque)
                
                if bloque_str in matriz_personal.index:
                    # Aplicar mínimo de cierre, pero respetar máximo efectivo
                    matriz_personal.loc[bloque_str, dia] = min(
                        max(matriz_personal.loc[bloque_str, dia], pers_cierre),
                        max_efectivo
                    )
        except Exception as e:
            continue
    
    return matriz_personal

# Calcular matrices de personal
matriz_personal_sala = calcular_personal_requerido(
    matriz_horas_sala, "SALA", local, dias_orden, 
    personal_apertura_sala, personal_cierre_sala
)

matriz_personal_cocina = calcular_personal_requerido(
    matriz_horas_cocina, "COCINA", local, dias_orden,
    personal_apertura_cocina, personal_cierre_cocina
)

# Calcular horas reales
horas_reales_sala = matriz_personal_sala.sum(axis=0) * 0.5
horas_reales_cocina = matriz_personal_cocina.sum(axis=0) * 0.5
horas_reales_totales = horas_reales_sala + horas_reales_cocina

# Calcular ventas por día
ventas_por_dia = matriz_ventas.sum(axis=0)

# Calcular productividad efectiva real
productividad_efectiva_real = ventas_por_dia / horas_reales_totales

# --------------------------------------------------
# OUTPUTS
# --------------------------------------------------
# Mostrar límites aplicados según local
if local == "LLURIA":
    limites_texto = "**Límites de personal para LLURIA:** Sala máx. 5 (4 L-V) | Cocina máx. 6 (5 L-V)"
else:
    limites_texto = f"**Límites de personal para {local}:** Sala máx. 4 (3 L-V) | Cocina máx. 5 (4 L-V)"

st.info(f"""
{limites_texto}

**Personal de Apertura/Cierre configurado:**
- **Sala:** {personal_apertura_sala} persona(s) apertura | {personal_cierre_sala} persona(s) cierre
- **Cocina:** {personal_apertura_cocina} persona(s) apertura | {personal_cierre_cocina} persona(s) cierre

*Nota: Se busca minimizar el personal manteniendo la operación eficiente.*
""")

st.markdown("---")
st.header("💰 Ventas diarias")
st.dataframe(ventas_df.round(2), use_container_width=True)

# --------------------------------------------------
# PRODUCTIVIDAD EFECTIVA
# --------------------------------------------------
st.markdown("---")
st.header("💼 Productividad Efectiva Detallada por Día")

st.info("""
**Productividad Efectiva** = Ventas del día / (Horas Reales Sala + Horas Reales Cocina)

Basado en personal requerido real con restricciones de apertura/cierre y límites máximos.
""")

productividad_df = pd.DataFrame({
    "Métrica": ["Ventas (€)", "Horas Sala", "Horas Cocina", "Horas Totales", "Productividad Efectiva (€/h)"],
    **{dia: [
        ventas_por_dia[dia],
        horas_reales_sala[dia],
        horas_reales_cocina[dia],
        horas_reales_totales[dia],
        productividad_efectiva_real[dia]
    ] for dia in dias_orden}
})

# Totales
total_ventas = ventas_por_dia.sum()
total_horas_sala = horas_reales_sala.sum()
total_horas_cocina = horas_reales_cocina.sum()
total_horas = horas_reales_totales.sum()
productividad_promedio = total_ventas / total_horas

productividad_df["TOTAL SEMANAL"] = [
    total_ventas, total_horas_sala, total_horas_cocina, total_horas, productividad_promedio
]

productividad_df["PROMEDIO SEMANAL"] = [
    total_ventas / 7, total_horas_sala / 7, total_horas_cocina / 7, total_horas / 7, productividad_promedio
]

def formatear_productividad(df):
    df_formatted = df.copy()
    for col in df.columns:
        if col == "Métrica":
            continue
        for idx in range(len(df)):
            metrica = df.loc[idx, "Métrica"]
            valor = df.loc[idx, col]
            if metrica in ["Ventas (€)", "Productividad Efectiva (€/h)"]:
                df_formatted.loc[idx, col] = f"€{valor:.2f}"
            else:
                df_formatted.loc[idx, col] = f"{valor:.2f}h"
    return df_formatted

st.dataframe(formatear_productividad(productividad_df), use_container_width=True)

# --------------------------------------------------
# DISTRIBUCIÓN DE VENTAS
# --------------------------------------------------
st.markdown("---")
st.header("💰 Distribución Promedio de Ventas Semanales")

fig_ventas = px.imshow(
    matriz_ventas,
    labels=dict(x="Día de la semana", y="Hora del día", color="Ventas (€)"),
    x=matriz_ventas.columns, y=matriz_ventas.index,
    color_continuous_scale="Greens", aspect="auto",
    title=f"Distribución Promedio de Ventas Semanales ({local})"
)

fig_ventas.update_layout(
    height=800, xaxis_title="Día de la semana", yaxis_title="Hora del día",
    yaxis=dict(autorange="reversed"), xaxis=dict(side="bottom")
)

fig_ventas.update_traces(
    text=matriz_ventas.round(0), texttemplate="€%{text}", textfont={"size": 7},
    hovertemplate='Día: %{x}<br>Hora: %{y}<br>Ventas: €%{z:.2f}<extra></extra>'
)

st.plotly_chart(fig_ventas, use_container_width=True)

# --------------------------------------------------
# DISTRIBUCIÓN DE PERSONAL
# --------------------------------------------------
st.markdown("---")
st.header("👥 Distribución de Personal Requerido por Bloques de 30 min")

st.info(f"""
**Restricciones aplicadas:**
- **Apertura:** {personal_apertura_sala} persona(s) en Sala y {personal_apertura_cocina} persona(s) en Cocina llegan 1 hora antes
- **Cierre:** {personal_cierre_sala} persona(s) en Sala y {personal_cierre_cocina} persona(s) en Cocina en los últimos bloques
- **{limites_texto}**
""")

area_personal = st.selectbox("Selecciona área", ["SALA", "COCINA"])
matriz_personal_area = matriz_personal_sala if area_personal == "SALA" else matriz_personal_cocina

fig_personal = px.imshow(
    matriz_personal_area,
    labels=dict(x="Día de la semana", y="Hora del día", color="Personal"),
    x=matriz_personal_area.columns, y=matriz_personal_area.index,
    color_continuous_scale="Blues", aspect="auto",
    title=f"Personal Requerido - {area_personal} ({local})"
)

fig_personal.update_layout(
    height=800, xaxis_title="Día de la semana", yaxis_title="Hora del día",
    yaxis=dict(autorange="reversed"), xaxis=dict(side="bottom")
)

fig_personal.update_traces(
    text=matriz_personal_area, texttemplate="%{text}", textfont={"size": 8},
    hovertemplate='Día: %{x}<br>Hora: %{y}<br>Personal: %{z}<extra></extra>'
)

st.plotly_chart(fig_personal, use_container_width=True)

# --------------------------------------------------
# EXPORTAR
# --------------------------------------------------
st.markdown("---")
st.subheader("💾 Exportar datos")

col1, col2, col3 = st.columns(3)

with col1:
    st.download_button("⬇️ Personal SALA", matriz_personal_sala.to_csv(index=True),
                      f"personal_sala_{local}.csv", "text/csv")
with col2:
    st.download_button("⬇️ Personal COCINA", matriz_personal_cocina.to_csv(index=True),
                      f"personal_cocina_{local}.csv", "text/csv")
with col3:
    st.download_button("⬇️ Productividad", productividad_df.to_csv(index=False),
                      f"productividad_{local}.csv", "text/csv")
