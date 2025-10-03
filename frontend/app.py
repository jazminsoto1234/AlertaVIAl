import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import joblib
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler

# =============================
# CONFIGURACIÓN DE STREAMLIT
# =============================
st.set_page_config(page_title="Micro-Paradas y Clustering", layout="wide")

st.title("🚦 Detección de Puntos Calientes de Congestión")
st.markdown("Sube un archivo de GPS (.csv o .xlsx) de un vehículo para visualizar sus micro-paradas y los clusters de congestión detectados por el modelo DBSCAN.")

# =============================
# CARGAR EL MODELO ENTRENADO
# =============================
# Se asume que el modelo 'modelo_dbscan1.joblib' está en la misma carpeta que este script.
@st.cache_resource
def load_model():
    try:
        model = joblib.load("../BD_Innovathon/modelo_dbscan1.joblib")
        return model
    except FileNotFoundError:
        st.error("Error Crítico: No se encontró el archivo del modelo 'modelo_dbscan1.joblib'. Asegúrate de que esté en la misma carpeta que el script.")
        return None

modelo = load_model()

# =============================
# FUNCIÓN 1: DETECTAR MICRO-PARADAS (Sin cambios)
# =============================
def detectar_microparadas(df: pd.DataFrame, col_vel: str) -> pd.DataFrame:
    df = df.copy()
    # Una micro-parada se define como una velocidad inferior a 5 km/h
    df["alerta"] = df[col_vel].apply(lambda v: "⚠️" if v < 5 else "✅")
    return df

# =============================
# FUNCIÓN 2: AGRUPAR CON MODELO (CORREGIDA)
# =============================
def agrupar_con_modelo(df: pd.DataFrame, col_vel: str) -> pd.DataFrame:
    df = df.copy()
    
    # Seleccionar las características que el modelo espera
    features_df = df[["Latitud", "Longitud", col_vel]]
    
    # Escalar los datos nuevos es crucial, ya que el modelo se entrenó con datos escalados.
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    
    # CORRECCIÓN CLAVE: DBSCAN no tiene un método .predict().
    # Para asignar clusters a nuevos datos, se debe usar .fit_predict().
    # Esto no re-entrena el modelo, sino que aplica las mismas reglas (eps, min_samples) a los nuevos datos.
    etiquetas = modelo.fit_predict(features_scaled)
    
    df["cluster"] = etiquetas
    return df

# =============================
# LÓGICA PRINCIPAL DE LA APLICACIÓN
# =============================
# Solo continuar si el modelo se cargó correctamente
if modelo:
    uploaded_file = st.file_uploader("Sube tu archivo de datos de GPS", type=["xlsx", "csv"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)

            # Limpieza de nombres de columnas
            df.columns = [c.strip().replace("\n", " ") for c in df.columns]
            st.success(f"Archivo '{uploaded_file.name}' cargado exitosamente.")

            # Detectar columna de velocidad
            col_vel_list = [c for c in df.columns if "velocidad" in c.lower()]
            if not col_vel_list:
                st.error("⚠️ No se encontró una columna de 'velocidad' en el archivo.")
                st.stop()
            col_vel = col_vel_list[0]
            
            # Validar que las columnas de coordenadas existan
            if "Latitud" not in df.columns or "Longitud" not in df.columns:
                 st.error("⚠️ El archivo debe contener las columnas 'Latitud' y 'Longitud'.")
                 st.stop()

            # --- Visualización 1: Micro-paradas ---
            st.subheader("🚗 Detección de Micro-Paradas (Velocidad < 5 km/h)")
            df_micro = detectar_microparadas(df, col_vel)

            m1 = folium.Map(location=[df_micro["Latitud"].mean(), df_micro["Longitud"].mean()], zoom_start=13)
            for _, row in df_micro.iterrows():
                color = "red" if row["alerta"] == "⚠️" else "green"
                popup = f"Velocidad: {row[col_vel]} km/h"
                folium.CircleMarker(location=[row["Latitud"], row["Longitud"]], radius=5, color=color, fill=True, fill_opacity=0.7, popup=popup).add_to(m1)
            st_folium(m1, use_container_width=True)

            # --- Visualización 2: Clustering con DBSCAN ---
            st.subheader("📊 Agrupación de Puntos Calientes con DBSCAN")
            df_cluster = agrupar_con_modelo(df, col_vel)

            m2 = folium.Map(location=[df_cluster["Latitud"].mean(), df_cluster["Longitud"].mean()], zoom_start=13)
            
            unique_clusters = sorted(df_cluster["cluster"].unique())
            color_map = cm.get_cmap('tab10', len(unique_clusters))
            cluster_colors = {c: mcolors.to_hex(color_map(i)) for i, c in enumerate(unique_clusters)}

            for _, row in df_cluster.iterrows():
                cluster_id = row["cluster"]
                color = "gray" if cluster_id == -1 else cluster_colors[cluster_id]
                popup = f"Cluster: {cluster_id} | Velocidad: {row[col_vel]} km/h"
                folium.CircleMarker(location=[row["Latitud"], row["Longitud"]], radius=5, color=color, fill=True, fill_opacity=0.7, popup=popup).add_to(m2)
            st_folium(m2, use_container_width=True)

            # --- Descarga de resultados ---
            st.subheader("📥 Descargar Resultados")
            df_final = df.copy()
            df_final["alerta_microparada"] = df_micro["alerta"]
            df_final["cluster_dbscan"] = df_cluster["cluster"]
            
            # Convertir a CSV para descargar (más universal)
            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar CSV con resultados", csv, "resultados_cluster.csv", "text/csv")

        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")

