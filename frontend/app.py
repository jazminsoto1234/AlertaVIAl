import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import joblib
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# =============================
# CONFIG STREAMLIT
# =============================
st.set_page_config(page_title="Micro-Paradas y Clustering", layout="wide")

st.title("🚦 Micro-Paradas y Agrupación de Movilidad")
st.markdown("Sube un archivo Excel/CSV con los registros de GPS para analizar micro-paradas y ver la agrupación con el modelo DBSCAN.")

# =============================
# CARGAR MODELO
# =============================
@st.cache_resource
def load_model():
    return joblib.load("/home/jazmin/Escritorio/AlertaVIAl/BD_Innovathon/modelo_dbscan1.joblib")

modelo = load_model()

# =============================
# FUNCIÓN 1: DETECTAR MICRO-PARADAS
# =============================
def detectar_microparadas(df: pd.DataFrame, col_vel: str) -> pd.DataFrame:
    df = df.copy()
    df["alerta"] = df[col_vel].apply(lambda v: "⚠️" if v < 5 else "✅")
    return df

# =============================
# FUNCIÓN 2: AGRUPAR CON MODELO
# =============================
def agrupar_con_modelo(df: pd.DataFrame, col_vel: str) -> pd.DataFrame:
    df = df.copy()
    features = df[["Latitud", "Longitud", col_vel]].values
    etiquetas = modelo.fit_predict(features)   # DBSCAN
    df["cluster"] = etiquetas
    return df

# =============================
# SUBIR ARCHIVO
# =============================
uploaded_file = st.file_uploader("Sube tu archivo", type=["xlsx", "csv"])

if uploaded_file:
    # Leer el Excel o CSV
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    # Normalizar nombres de columnas
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("\n", " ")

    st.success(f"Archivo cargado: {uploaded_file.name}")
    st.write("Columnas detectadas:", df.columns.tolist())

    # Detectar columna de velocidad automáticamente
    try:
        col_vel = [c for c in df.columns if "velocidad" in c.lower()][0]
    except IndexError:
        st.error("⚠️ No se encontró ninguna columna de velocidad en el archivo.")
        st.stop()

    # --- Micro-paradas ---
    st.subheader("🚗 Detección de Micro-Paradas (Regla de Velocidad)")
    df_micro = detectar_microparadas(df, col_vel)

    m1 = folium.Map(location=[df_micro["Latitud"].mean(), df_micro["Longitud"].mean()], zoom_start=14)
    for _, row in df_micro.iterrows():
        color = "red" if row["alerta"] == "⚠️" else "green"
        popup = f"{col_vel}: {row[col_vel]} km/h | Alerta: {row['alerta']}"
        folium.CircleMarker(
            location=[row["Latitud"], row["Longitud"]],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=popup
        ).add_to(m1)
    st_folium(m1, width=900, height=500)

    # --- Clustering con modelo ---
    st.subheader("📊 Agrupación con DBSCAN")
    df_cluster = agrupar_con_modelo(df, col_vel)

    m2 = folium.Map(location=[df_cluster["Latitud"].mean(), df_cluster["Longitud"].mean()], zoom_start=14)

    unique_clusters = sorted(set(df_cluster["cluster"]))
    color_map = cm.get_cmap('tab10', len(unique_clusters))
    cluster_colors = {c: mcolors.to_hex(color_map(i)) for i, c in enumerate(unique_clusters)}

    for _, row in df_cluster.iterrows():
        cluster = row["cluster"]
        color = "gray" if cluster == -1 else cluster_colors[cluster]
        popup = f"Cluster: {cluster} | {col_vel}: {row[col_vel]}"
        folium.CircleMarker(
            location=[row["Latitud"], row["Longitud"]],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=popup
        ).add_to(m2)
    st_folium(m2, width=900, height=500)

    # --- Descarga resultados ---
    st.subheader("📥 Descargar resultados")
    out_name = "resultados_microparadas_cluster.xlsx"
    df_final = df_micro.copy()
    df_final["cluster"] = df_cluster["cluster"]
    df_final.to_excel(out_name, index=False)
    with open(out_name, "rb") as f:
        st.download_button("Descargar Excel procesado", f, file_name=out_name)
