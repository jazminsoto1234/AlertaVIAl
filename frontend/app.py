import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# =============================
# CONFIG STREAMLIT
# =============================
st.set_page_config(page_title="Detección de Micro-Paradas", layout="wide")

st.title("🚦 Detección de Micro-Paradas y Predicción de Colas")
st.markdown("Sube un archivo Excel/CSV con los registros de GPS para analizar micro-paradas.")

# =============================
# FUNCIÓN DE PROCESAMIENTO
# =============================
def detectar_microparadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aquí conectas tu modelo entrenado.
    Por ahora, simulamos que si la velocidad < 5 km/h => alerta.
    """
    df = df.copy()
    df["alerta"] = df["Velocidad (km/h)"].apply(lambda v: "⚠️" if v < 5 else "✅")
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

    st.success(f"Archivo cargado: {uploaded_file.name}")
    st.write("Vista previa de los datos:")
    st.dataframe(df.head())

    # Procesar con tu modelo
    df_proc = detectar_microparadas(df)

    # =============================
    # MOSTRAR MAPA
    # =============================
    st.subheader("📍 Mapa de Micro-Paradas")

    if "Latitud" in df_proc.columns and "Longitud" in df_proc.columns:
        # Centro del mapa = promedio de coordenadas
        m = folium.Map(location=[df_proc["Latitud"].mean(), df_proc["Longitud"].mean()], zoom_start=14)

        for _, row in df_proc.iterrows():
            color = "red" if row["alerta"] == "⚠️" else "green"
            popup = f"Velocidad: {row['Velocidad (km/h)']} km/h | Alerta: {row['alerta']}"
            folium.CircleMarker(
                location=[row["Latitud"], row["Longitud"]],
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=popup
            ).add_to(m)

        st_folium(m, width=900, height=500)
    else:
        st.error("⚠️ El archivo debe contener columnas 'Latitud' y 'Longitud'.")

    # =============================
    # DESCARGAR RESULTADOS
    # =============================
    st.subheader("📥 Descargar resultados")
    out_name = "resultados_microparadas.xlsx"
    df_proc.to_excel(out_name, index=False)
    with open(out_name, "rb") as f:
        st.download_button("Descargar Excel procesado", f, file_name=out_name)
