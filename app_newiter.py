# app_newiter.py — UI principal con Streamlit
import os
from datetime import date, time

import pandas as pd
import streamlit as st
import folium
from folium.plugins import Geocoder
from streamlit_folium import st_folium

# ---- Ajuste CSS para quitar whitespace extra de st_folium
st.markdown(
    """
    <style>
    /* Quita márgenes/padding alrededor de los mapas */
    iframe[title="st_folium"] {
        display: block;
        margin: 0 auto !important;
        padding: 0 !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(iframe[title="st_folium"]) {
        margin-bottom: 0rem !important;
        padding-bottom: 0rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


from core import (
    load_events, save_events,
    save_daystart, load_daystart,
    lookup_barrio_y_valor, valor_reponderado,
    build_nodes_and_matrix, GreedyTWOptimizer, run_greedy_optimizer
)

EVENTS_CSV = "events_for_optimizer.csv"

# -----------------------------------------------------------
# Utilidades
# -----------------------------------------------------------
NEEDED_COLS = [
    "event_id","name","earliest_start","latest_start","duration_min",
    "lat","lon", "scale","location_score","demo_18_29","demo_30_44","demo_45_64","demo_65p",
    "reach","must_visit","barrio_id","barrio_valor_base","value"
]

def ensure_events_file():
    need_init = (not os.path.exists(EVENTS_CSV)) or os.path.getsize(EVENTS_CSV) == 0
    if need_init:
        df0 = pd.DataFrame(columns=NEEDED_COLS)
        df0.to_csv(EVENTS_CSV, index=False)

def next_event_id_safe():
    df = load_events()
    if df.empty or "event_id" not in df.columns:
        return 1
    max_id = pd.to_numeric(df["event_id"], errors="coerce").dropna().max()
    return 1 if pd.isna(max_id) else int(max_id) + 1


def map_picker(key: str, center=(-33.45, -70.65), zoom=12, height=430):
    """Mapa con buscador: captura clics o marcador del geocoder"""
    lat_key, lon_key = f"{key}_lat", f"{key}_lon"
    st.session_state.setdefault(lat_key, None)
    st.session_state.setdefault(lon_key, None)

    m = folium.Map(location=center, zoom_start=zoom, control_scale=True)
    Geocoder(collapsed=False, add_marker=True, position="topleft").add_to(m)
    folium.LatLngPopup().add_to(m)

    if st.session_state[lat_key] and st.session_state[lon_key]:
        folium.Marker(
            [st.session_state[lat_key], st.session_state[lon_key]],
            tooltip="Seleccionado"
        ).add_to(m)

    out = st_folium(m, height=250, width=720, key=key)

    # 1) Click manual
    if out and out.get("last_clicked"):
        st.session_state[lat_key] = out["last_clicked"]["lat"]
        st.session_state[lon_key] = out["last_clicked"]["lng"]

    # 2) Pin del geocoder → viene en all_markers
    elif out and out.get("all_markers"):
        last = out["all_markers"][-1]
        st.session_state[lat_key] = last.get("lat") or last.get("location", {}).get("lat")
        st.session_state[lon_key] = last.get("lng") or last.get("location", {}).get("lng")

    sel_lat = st.session_state[lat_key]
    sel_lon = st.session_state[lon_key]
    if sel_lat and sel_lon:
        st.caption(f"🧭 Seleccionado: {sel_lat:.5f}, {sel_lon:.5f}")
    else:
        st.caption("Haz una búsqueda o un click en el mapa para seleccionar.")
    return sel_lat, sel_lon


def combine_today(t: time) -> pd.Timestamp:
    return pd.Timestamp.combine(date.today(), t)

# -----------------------------------------------------------
# App
# -----------------------------------------------------------
st.set_page_config(page_title="Scheduler UI", layout="wide")
ensure_events_file()

st.title("📍 Punto de inicio del día")

# ----- Punto de inicio -----
lat_start, lon_start = map_picker(key="map_start")

col_a, col_b = st.columns([1,2])
with col_a:
    if st.button("💾 Confirmar y guardar inicio"):
        if lat_start is not None and lon_start is not None:
            save_daystart(float(lat_start), float(lon_start))
            st.success("Inicio guardado.")
        else:
            st.error("Primero selecciona un punto en el mapa.")

st.markdown("---")

# ----- Cargar evento -----
st.header("🗂️ Cargar evento")

# 🔸 Mapa fuera del form, para que capture el pin del geocoder o el clic
st.subheader("Ubicación del evento")
lat_ev, lon_ev = map_picker(key="map_event", center=(-33.45, -70.65), zoom=12, height=430)

with st.form("add_event_form_v1", clear_on_submit=True):  # ✅ key único
    name = st.text_input("Nombre del evento")
    c1, c2, c3 = st.columns(3)
    with c1:
        earliest = st.time_input("Inicio más temprano", time(9, 0))
    with c2:
        latest = st.time_input("Inicio más tardío", time(18, 0))
    with c3:
        duration_min = st.number_input("Duración (min)", min_value=5, max_value=480, value=60, step=5)

    must_visit = st.checkbox("Debe visitarse sí o sí", value=False)

    # 🔹 Nuevo: escala del evento
    scale = st.selectbox(
        "Escala del evento (personas esperadas)",
        options=[10, 100, 1000],
        index=1  # por defecto 100
    )

    submitted = st.form_submit_button("➕ Guardar evento")
    if submitted:
        if not name.strip():
            st.error("Escribe un nombre para el evento.")
        elif lat_ev is None or lon_ev is None:
            st.error("Selecciona la ubicación en el mapa (buscar o click).")
        else:
            barrio_id, base_val = lookup_barrio_y_valor(float(lat_ev), float(lon_ev))
            value = valor_reponderado(base_val, 0, 0, 0, 0) * int(scale)

            new_event = {
                "event_id": next_event_id_safe(),
                "name": name.strip(),
                "earliest_start": combine_today(earliest),
                "latest_start": combine_today(latest),
                "duration_min": int(duration_min),
                "lat": float(lat_ev),
                "lon": float(lon_ev),
                "scale": int(scale),  # 👈 ahora guardamos escala
                "location_score": pd.NA,
                "demo_18_29": pd.NA, "demo_30_44": pd.NA, "demo_45_64": pd.NA, "demo_65p": pd.NA,
                "reach": pd.NA,
                "must_visit": bool(must_visit),
                "barrio_id": barrio_id,
                "barrio_valor_base": base_val,
                "value": value,
            }

            df = load_events()
            df = pd.concat([df, pd.DataFrame([new_event])], ignore_index=True)
            save_events(df)
            st.success(f"Evento '{name}' guardado (barrio: {barrio_id}, escala: {scale}).")

# ----- Listado de eventos -----
st.subheader("📋 Eventos cargados")
df_events = load_events()
if df_events.empty:
    st.info("No hay eventos cargados aún.")
else:
    st.dataframe(
        df_events[
            ["event_id","name","earliest_start","latest_start","duration_min",
             "lat","lon","scale","must_visit","barrio_id","barrio_valor_base","value"]
        ].sort_values("event_id"),
        use_container_width=True
    )

st.markdown("---")

from core import enrich_events, to_minutes_since


# ---- helper para formato HH:MM
def fmt_hhmm(day0, minutes):
    if minutes is None or pd.isna(minutes):
        return "-"
    return (day0 + pd.Timedelta(minutes=int(minutes))).strftime("%H:%M")

# ----- Optimizador -----
st.header("🛠️ Optimización de ruta")

# espacio en sesión para persistir resultados
if "opt_result" not in st.session_state:
    st.session_state["opt_result"] = None

def _fmt_hhmm(day0, minutes):
    try:
        return (day0 + pd.Timedelta(minutes=int(minutes))).strftime("%H:%M")
    except Exception:
        return "-"

if st.button("⚡ Optimizar ruta"):
    try:
        df_events = load_events()
        if df_events.empty:
            st.error("No hay eventos para optimizar.")
        else:
            # Re-enriquecer + normalizar
            from core import enrich_events, to_minutes_since  # por si no estaba arriba
            df_events = enrich_events(df_events)

            dep = load_daystart()
            if dep is None or dep[0] is None or dep[1] is None:
                st.error("Primero define y guarda el punto de inicio del día.")
            else:
                start_lat, start_lon = float(dep[0]), float(dep[1])
                day0 = pd.to_datetime(df_events["earliest_start"].min()).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                end_min = to_minutes_since(
                    day0,
                    pd.to_datetime(df_events["latest_start"].max()).to_pydatetime()
                )

                # Nodos + matriz
                nodes, T = build_nodes_and_matrix(
                    df_events, day0,
                    start_min=0, end_min=end_min,
                    start_lat=start_lat, start_lon=start_lon
                )

                # Optimizar
                opt = GreedyTWOptimizer(df_events, T, nodes, lambda_travel=0.1, day=day0)
                route, sched, stats = opt.solve_with_must()

                # Guardar TODO lo necesario en sesión para el render persistente
                coords = [(opt.id2node[n]["lat"], opt.id2node[n]["lon"]) for n in route]
                names  = [opt.id2node[n]["name"] for n in route]
                if coords:
                    center = [sum(c[0] for c in coords)/len(coords),
                              sum(c[1] for c in coords)/len(coords)]
                else:
                    center = [-33.45, -70.65]

                st.session_state["opt_result"] = {
                    "route": route,
                    "sched": sched,
                    "opt": opt,
                    "day0": day0,
                    "coords": coords,
                    "names": names,
                    "center": center,
                }

                st.success("Optimización lista ✅")
    except Exception as e:
        st.session_state["opt_result"] = None
        st.error(f"Error al optimizar: {e}")

# ---- Render persistente de resultados (si existen)
res = st.session_state.get("opt_result")
if res:
    # 1) Extraer objetos guardados
    route = res.get("route", [])
    sched = res.get("sched", {})
    opt   = res.get("opt", None)
    day0  = res.get("day0", pd.Timestamp.today().normalize())

    # 2) Construir tabla con horas HH:MM y ordenar cronológicamente
    st.subheader("📋 Horario optimizado")

    rows = []
    for node_id in route:
        node = opt.id2node[node_id]
        if node_id in sched:
            r = sched[node_id]
            rows.append({
                "Nodo": node["name"],
                "Inicio": _fmt_hhmm(day0, r["start"]),
                "Fin": _fmt_hhmm(day0, r["depart"]),
            })
        else:
            rows.append({"Nodo": node["name"], "Inicio": "•", "Fin": "•"})

    df_sched = pd.DataFrame(rows)

    # Orden cronológico por inicio (pone "•" al final)
    def _parse_time_to_sort(x):
        try:
            return pd.to_datetime(x, format="%H:%M")
        except Exception:
            return pd.NaT

    df_sched["_sort"] = df_sched["Inicio"].apply(_parse_time_to_sort)
    df_sched = df_sched.sort_values("_sort", na_position="last").drop(columns="_sort")

    st.table(df_sched)

    # 3) Mapa de ruta
    st.subheader("🗺️ Ruta en el mapa")
    if res.get("coords"):
        coords = res["coords"]
        names  = res["names"]

        m = folium.Map(location=res["center"], zoom_start=12)

        # Marcadores con nombres
        for idx, (lat, lon) in enumerate(coords):
            folium.Marker(
                [lat, lon],
                tooltip=f"{idx}. {names[idx]}",
                icon=folium.Icon(
                    color=("blue" if 0 < idx < len(coords)-1 else "green"),
                    icon="info-sign"
                )
            ).add_to(m)

        # Intentar trazar por calles con OSRM
        try:
            import requests
            url = "http://router.project-osrm.org/route/v1/driving/"
            coord_str = ";".join([f"{lon},{lat}" for lat, lon in coords])
            r = requests.get(url + coord_str, params={"overview": "full", "geometries": "geojson"}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data.get("routes"):
                    geometry = data["routes"][0]["geometry"]
                    folium.GeoJson(geometry, name="route").add_to(m)
                else:
                    st.warning("No se pudo calcular la ruta por calles, se dibuja línea recta.")
                    folium.PolyLine(coords, weight=4, opacity=0.8, color="red").add_to(m)
            else:
                st.warning("OSRM no respondió, usando línea recta.")
                folium.PolyLine(coords, weight=4, opacity=0.8, color="red").add_to(m)
        except Exception as e:
            st.warning(f"No se pudo usar OSRM: {e}")
            folium.PolyLine(coords, weight=4, opacity=0.8, color="red").add_to(m)

        st_folium(m, width=720, height=480, key="map_result")
    else:
        st.info("No hay coordenadas para dibujar la ruta.")
