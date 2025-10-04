import os
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
import joblib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from twilio.rest import Client

# =============================
# CONFIGURACIÓN DE STREAMLIT
# =============================
st.set_page_config(page_title="Micro-Paradas y Clustering", layout="wide")
st.title("🚦 Detección de Puntos Calientes de Congestión")
st.markdown(
    "Sube un archivo de GPS (.csv o .xlsx) para visualizar sus micro-paradas "
    "y los clusters de congestión detectados por DBSCAN."
)

# =============================
# CARGAR EL MODELO ENTRENADO
# =============================
@st.cache_resource
def load_model():
    try:
        # Ajusta la ruta si tu .joblib está en otra carpeta
        model = joblib.load("../BD_Innovathon/modelo_dbscan1.joblib")
        return model
    except FileNotFoundError:
        st.error("❌ No se encontró '../BD_Innovathon/modelo_dbscan1.joblib'. Verifica la ruta.")
        return None

modelo = load_model()

# =============================
# UTIL: MICRO-PARADAS
# =============================
def detectar_microparadas(df: pd.DataFrame, col_vel: str) -> pd.DataFrame:
    df = df.copy()
    df["alerta"] = df[col_vel].apply(lambda v: "⚠️" if v < 5 else "✅")
    return df

# =============================
# UTIL: AGRUPAR CON DBSCAN
# =============================
def agrupar_con_modelo(df: pd.DataFrame, col_vel: str, model) -> pd.DataFrame:
    df = df.copy()
    features_df = df[["Latitud", "Longitud", col_vel]].astype(float)
    # Escala por consistencia con el entrenamiento
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df)
    # Para DBSCAN usamos fit_predict (aplica eps/min_samples cargados)
    etiquetas = model.fit_predict(X)
    df["cluster"] = etiquetas
    return df

# =============================
# UTIL: TWILIO (lee secrets/env)
# =============================
def get_twilio_client_and_from():
    # 1) Intenta estructura seccionada: [twilio]
    try:
        sid = st.secrets["twilio"]["account_sid"]
        token = st.secrets["twilio"]["auth_token"]
        from_num = st.secrets["twilio"]["from_number"]
    except Exception:
        # 2) Fallback a claves planas o variables de entorno
        sid = st.secrets.get("TWILIO_ACCOUNT_SID", os.getenv("TWILIO_ACCOUNT_SID"))
        token = st.secrets.get("TWILIO_AUTH_TOKEN", os.getenv("TWILIO_AUTH_TOKEN"))
        from_num = st.secrets.get("TWILIO_FROM_NUMBER", os.getenv("TWILIO_FROM_NUMBER"))

    if not (sid and token and from_num):
        return None, None
    return Client(sid, token), from_num

def get_default_to_number():
    try:
        return st.secrets["twilio"]["to_number"]
    except Exception:
        return os.getenv("TWILIO_TO_NUMBER", "")

def send_sms(body: str, to_number: str):
    client, from_number = get_twilio_client_and_from()
    if not (client and from_number and to_number):
        raise RuntimeError("Credenciales/número de Twilio incompletos.")
    msg = client.messages.create(body=body, from_=from_number, to=to_number)
    return msg.sid

# =============================
# APP
# =============================
if modelo is None:
    st.stop()

uploaded_file = st.file_uploader("Sube tu archivo de datos de GPS", type=["xlsx", "csv"])

# --- Panel de Notificaciones por SMS (config) ---
with st.expander("🔔 Notificación por SMS (Twilio)"):
    sms_enable = st.checkbox("Enviar SMS cuando haya un cluster grande", value=False)
    min_cluster_size = st.number_input(
        "Tamaño mínimo del cluster para avisar",
        min_value=3, max_value=500, value=10, step=1
    )
    to_number = st.text_input("Número destino (E.164, ej. +519xxxxxxxx)", value=get_default_to_number())
    usar_filtro_velocidad = st.checkbox("Además, alertar solo si la velocidad promedio del cluster < 5 km/h", value=False)

if not uploaded_file:
    st.info("📄 Sube un archivo para comenzar.")
    st.stop()

# ---------- Lectura ----------
try:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"❌ Error leyendo el archivo: {e}")
    st.stop()

# Limpieza de columnas
df.columns = [c.strip().replace("\n", " ") for c in df.columns]
st.success(f"Archivo **{uploaded_file.name}** cargado correctamente.")

# Detectar columna de velocidad
cands_vel = [c for c in df.columns if "velocidad" in c.lower()]
if not cands_vel:
    st.error("⚠️ No se encontró columna de velocidad. Renombra/asegura que exista.")
    st.stop()
col_vel = cands_vel[0]

# Validar coordenadas
if "Latitud" not in df.columns or "Longitud" not in df.columns:
    st.error("⚠️ El archivo debe contener las columnas 'Latitud' y 'Longitud'.")
    st.stop()

# ---------- Micro-paradas ----------
st.subheader("🚗 Micro-Paradas (Velocidad < 5 km/h)")
df_micro = detectar_microparadas(df, col_vel)

m1 = folium.Map(location=[df_micro["Latitud"].mean(), df_micro["Longitud"].mean()], zoom_start=13)
for _, row in df_micro.iterrows():
    color = "red" if row["alerta"] == "⚠️" else "green"
    popup = f"Velocidad: {row[col_vel]} km/h"
    folium.CircleMarker(
        location=[row["Latitud"], row["Longitud"]],
        radius=5, color=color, fill=True, fill_opacity=0.7, popup=popup
    ).add_to(m1)
st_folium(m1, use_container_width=True)

# ---------- DBSCAN ----------
st.subheader("📊 Agrupación de Puntos Calientes con DBSCAN")
df_cluster = agrupar_con_modelo(df, col_vel, modelo)

m2 = folium.Map(location=[df_cluster["Latitud"].mean(), df_cluster["Longitud"].mean()], zoom_start=13)
unique_clusters = sorted(df_cluster["cluster"].unique())
color_map = cm.get_cmap('tab10', max(1, len(unique_clusters)))
cluster_colors = {c: mcolors.to_hex(color_map(i)) for i, c in enumerate(unique_clusters)}

for _, row in df_cluster.iterrows():
    c_id = row["cluster"]
    color = "gray" if c_id == -1 else cluster_colors[c_id]
    popup = f"Cluster: {c_id} | Velocidad: {row[col_vel]} km/h"
    folium.CircleMarker(
        location=[row["Latitud"], row["Longitud"]],
        radius=5, color=color, fill=True, fill_opacity=0.7, popup=popup
    ).add_to(m2)
st_folium(m2, use_container_width=True)

# ---------- Resumen + Notificación ----------
valid = df_cluster[df_cluster["cluster"] != -1]
summary = (
    valid.groupby("cluster", as_index=False)
         .agg(count=("cluster", "size"),
              lat=("Latitud", "mean"),
              lon=("Longitud", "mean"),
              vel_prom=(col_vel, "mean"))
         .sort_values("count", ascending=False)
)
st.subheader("📋 Resumen de clusters")
st.dataframe(summary)

# Trigger SMS
if sms_enable and not summary.empty:
    filt = summary["count"] >= min_cluster_size
    if usar_filtro_velocidad:
        filt &= (summary["vel_prom"] < 5)

    grandes = summary[filt]
    if grandes.empty:
        st.info("No hay clusters que cumplan las condiciones para alertar.")
    else:
        try:
            enviados = 0
            for _, r in grandes.iterrows():
                msg = (
                    f"🚦 Alerta DBSCAN: cluster #{int(r['cluster'])} detectado con "
                    f"{int(r['count'])} puntos. Centro aprox: ({r['lat']:.5f}, {r['lon']:.5f})"
                    + (f", vel prom: {r['vel_prom']:.1f} km/h" if usar_filtro_velocidad else "")
                )
                sid = send_sms(msg, to_number)
                enviados += 1
            st.success(f"SMS enviado(s) para {enviados} cluster(s) que cumplen el criterio.")
        except Exception as e:
            st.error(f"❌ Error enviando SMS: {e}")

# ---------- Descarga ----------
st.subheader("📥 Descargar resultados")
df_final = df.copy()
df_final["alerta_microparada"] = df_micro["alerta"]
df_final["cluster_dbscan"] = df_cluster["cluster"]
csv = df_final.to_csv(index=False).encode("utf-8")
st.download_button("Descargar CSV con resultados", csv, "resultados_cluster.csv", "text/csv")