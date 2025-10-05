# core.py — lógica central del optimizador y manejo de eventos/barrios
import os
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import re
import gdown




# Geoespacial (opcional)
try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:
    gpd = None
    Point = None

# ---------------------------------------------------------------------------
# Archivos / rutas
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVENTS_CSV = os.path.join(BASE_DIR, "events_for_optimizer.csv")
OUT_DIR = os.path.join(BASE_DIR, "out")
BARRIOS_FILE = os.path.join(BASE_DIR, "barrios_nacional.gpkg")
VALORES_EXCEL = os.path.join(BASE_DIR, "Valor_Barrios.xlsx")
DAYSTART_CSV = os.path.join(BASE_DIR, "daystart_coordinates.csv")

os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"  # permitir features grandes

# ---------------------------------------------------------------------------
# Esquema esperado
# ---------------------------------------------------------------------------
# columnas requeridas globales
NEEDED_COLS = [
    "event_id","name","earliest_start","latest_start","duration_min",
    "lat","lon","location_score",
    "demo_18_29","demo_30_44","demo_45_64","demo_65p",
    "reach","must_visit","barrio_id","barrio_valor_base","scale","value"
]

def _ensure_events_file():
    """Crea el archivo de eventos vacío si no existe"""
    if not os.path.exists(EVENTS_CSV):
        pd.DataFrame(columns=NEEDED_COLS).to_csv(EVENTS_CSV, index=False)

def load_events() -> pd.DataFrame:
    _ensure_events_file()
    try:
        df = pd.read_csv(EVENTS_CSV, parse_dates=["earliest_start","latest_start"])
    except Exception:
        return pd.DataFrame(columns=NEEDED_COLS)

    # valores por defecto
    defaults = {
        "location_score": np.nan,
        "demo_18_29": 0.0, "demo_30_44": 0.0, "demo_45_64": 0.0, "demo_65p": 0.0,
        "reach": np.nan,
        "must_visit": False,
        "barrio_id": "",
        "barrio_valor_base": np.nan,
        "scale": 1.0,
        "value": np.nan,
    }
    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v

    # asegurar event_id
    if "event_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["event_id"] = df.index + 1

    df["must_visit"] = df["must_visit"].astype(bool)

    # asegurar columnas ordenadas
    for col in NEEDED_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[NEEDED_COLS]

    return df


def save_events(df: pd.DataFrame):
    df.to_csv(EVENTS_CSV, index=False)

def save_event(row: dict):
    """Guarda un evento nuevo en events_for_optimizer.csv asegurando el esquema completo."""
    _ensure_events_file()
    df = load_events()

    # asegurar que row tenga todas las columnas
    for col in NEEDED_COLS:
        if col not in row:
            if col == "event_id":
                row[col] = next_event_id()
            elif col == "scale":
                row[col] = 1.0
            elif col == "must_visit":
                row[col] = False
            else:
                row[col] = np.nan

    # concatenar y reordenar columnas
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df[NEEDED_COLS]
    df.to_csv(EVENTS_CSV, index=False)


def next_event_id() -> int:
    df = load_events()
    return 1 if df.empty else int(df["event_id"].max()) + 1

# ---------------------------------------------------------------------------
# Tiempo
# ---------------------------------------------------------------------------
def to_minutes_since(day: datetime, dt: datetime) -> int:
    return int((dt - day).total_seconds() // 60)

def as_dt(day: datetime, minutes: int) -> datetime:
    return day + timedelta(minutes=int(minutes))

# ---------------------------------------------------------------------------
# Día inicial (punto de inicio)
# ---------------------------------------------------------------------------
def save_daystart(lat: float, lon: float, path: str = DAYSTART_CSV):
    """Guardar las coordenadas del punto de inicio en CSV (sobrescribe)."""
    pd.DataFrame([{"lat": float(lat), "lon": float(lon)}]).to_csv(path, index=False)

def load_daystart(path: str = DAYSTART_CSV):
    """Cargar coordenadas del punto de inicio desde CSV. Devuelve (lat, lon) o (None, None)."""
    if not os.path.exists(path):
        return None, None
    try:
        df = pd.read_csv(path)
        if not df.empty and {"lat","lon"}.issubset(df.columns):
            return float(df.iloc[0]["lat"]), float(df.iloc[0]["lon"])
    except Exception:
        pass
    return None, None

# ---------------------------------------------------------------------------
# Barrios y valores
# ---------------------------------------------------------------------------
def load_barrios_merged():
    """
    Loads the merged barrios dataset, downloading it from Google Drive if needed.
    Works in both local and Streamlit Cloud environments.
    """
    import os
    import tempfile
    import pandas as pd
    import geopandas as gpd
    import gdown

    file_id = "1x5LfDYpYlpgeYsyIKmuCdJGruKvULyC8"
    url = f"https://drive.google.com/uc?id={file_id}"

    tmp_path = os.path.join(tempfile.gettempdir(), "barrios_nacional.gpkg")

    # --- Download if needed ---
    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 1e6:
        print("⬇️ Downloading barrios_nacional.gpkg from Google Drive...")
        gdown.download(url, tmp_path, quiet=False)

    # --- Load the geopackage ---
    print("📥 Reading barrios_nacional.gpkg...")
    gdf = gpd.read_file(tmp_path)

    print("🔍 Barrios columns:", gdf.columns.tolist())
    print("🧮 Row count:", len(gdf))

    # --- Load Excel values ---
    valores = pd.read_excel("Valor_Barrios.xlsx")
    valores.columns = [c.strip().lower() for c in valores.columns]
    print("📗 Valor_Barrios columns:", valores.columns.tolist())

    # --- Detect and normalize join keys ---
    barrio_col_gdf = None
    for cand in ["barrio_id", "barrioid", "id_barrio", "barrio"]:
        if cand in gdf.columns:
            barrio_col_gdf = cand
            break
    if not barrio_col_gdf:
        print("⚠️ Could not find barrio column in GeoPackage.")
        barrio_col_gdf = gdf.columns[0]

    barrio_col_val = None
    for cand in ["barrio_id", "barrioid", "id_barrio", "barrio"]:
        if cand in valores.columns:
            barrio_col_val = cand
            break
    if not barrio_col_val:
        print("⚠️ Could not find barrio column in Valor_Barrios.xlsx.")
        barrio_col_val = valores.columns[0]

    print(f"🔗 Joining on gdf['{barrio_col_gdf}'] and valores['{barrio_col_val}'].")

    # --- Merge ---
    merged = gdf.merge(valores, left_on=barrio_col_gdf, right_on=barrio_col_val, how="left")

    print("✅ Merge complete. Rows:", len(merged))
    print("🧩 Sample merged columns:", merged.columns.tolist()[:10])

    sidx = merged.sindex if hasattr(merged, "sindex") else None
    return merged, sidx, None






BARRIOS_GDF, BARRIOS_SIDX, BARRIOS_ERR = load_barrios_merged()

def lookup_barrio_y_valor(lat: float, lon: float):
    if BARRIOS_GDF is None or BARRIOS_ERR or BARRIOS_GDF.empty:
        print("⚠️  No barrios data loaded:", BARRIOS_ERR)
        return "Desconocido", 0.0

    print(f"🔍 Checking point ({lat}, {lon}) against {len(BARRIOS_GDF)} polygons")
    pt = Point(lon, lat)

    cand = BARRIOS_GDF
    if BARRIOS_SIDX is not None:
        cand_idx = list(BARRIOS_SIDX.intersection((pt.x, pt.y, pt.x, pt.y)))
        print(f"Candidate polygons from spatial index: {len(cand_idx)}")
        if not cand_idx:
            return "Fuera de barrio", 0.0
        cand = BARRIOS_GDF.iloc[cand_idx]

    hit = cand[cand.contains(pt)]
    print(f"Polygons containing point: {len(hit)}")

    if hit.empty:
        return "Fuera de barrio", 0.0

    h = hit.iloc[0]
    barrio_id = h.get("barrio_id", "Desconocido")
    base_val = float(h.get("puntaje_absoluto", 0.0))
    return barrio_id, base_val


def valor_reponderado(base_val: float,
                      demo18: float, demo30: float, demo45: float, demo65: float,
                      scale: float = 1.0) -> float:
    """
    Calcula el valor reponderado del evento en función de:
      - Valor base del barrio
      - Distribución demográfica
      - Escala del evento
    """
    demo = np.array([demo18, demo30, demo45, demo65], dtype=float)
    demo = np.nan_to_num(demo, nan=0.0)

    if np.allclose(demo.sum(), 0.0):
        factor = 1.0
    else:
        w = np.array([0.40, 0.35, 0.20, 0.05], dtype=float)
        factor = float(demo @ w) * 4.0

    return round(float(base_val) * factor * float(scale), 2)


# ---------------------------------------------------------------------------
# Distancias
# ---------------------------------------------------------------------------
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    p = math.pi/180
    dphi = (lat2-lat1)*p
    dlambda = (lon2-lon1)*p
    a = math.sin(dphi/2)**2 + math.cos(lat1*p)*math.cos(lat2*p)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))

def minutes_haversine(lat1, lon1, lat2, lon2, speed_kmh=25.0) -> int:
    d_km = haversine_m(lat1, lon1, lat2, lon2) / 1000.0
    travel = (d_km / max(1e-6, speed_kmh)) * 60.0
    return max(1, int(round(travel)))

def build_travel_matrix(coords, speed_kmh=25.0):
    n = len(coords)
    mat = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i, j] = minutes_haversine(*coords[i], *coords[j], speed_kmh=speed_kmh)
    return mat

# ---------------------------------------------------------------------------
# Optimizador
# ---------------------------------------------------------------------------
class GreedyTWOptimizer:
    def __init__(self, events_df, travel_minutes, nodes, lambda_travel, day):
        self.events = events_df.copy()
        self.T = np.asarray(travel_minutes, dtype=int)
        self.nodes = nodes
        self.id2node = {n["idx"]: n for n in nodes}
        self.idx = {n["idx"]: k for k, n in enumerate(nodes)}
        self.N = len(events_df)
        self.lmb = float(lambda_travel)
        self.val = self.events.set_index("event_id")["value"].to_dict()
        self.must = set(self.events.query("must_visit == True")["event_id"].astype(int).tolist())

    def _schedule(self, route):
        sched = {}
        t = self.id2node[route[0]]["a"]
        for r in range(len(route)-1):
            i, j = route[r], route[r+1]
            travel = int(self.T[self.idx[i], self.idx[j]])
            arrive_j = t + int(self.id2node[i]["duration"]) + travel
            a_j, b_j = int(self.id2node[j]["a"]), int(self.id2node[j]["b"])
            start_j = max(arrive_j, a_j)
            if start_j > b_j:
                return False, {}
            sched[j] = {"arrive": arrive_j, "start": start_j,
                        "depart": start_j + int(self.id2node[j]["duration"])}
            t = start_j
        return True, sched

    def _route_value(self, route):
        ev_ids = [i for i in route if i not in (0, self.N+1)]
        value = sum(float(self.val.get(i, 0.0)) for i in ev_ids)
        travel = sum(int(self.T[self.idx[a], self.idx[b]]) for a, b in zip(route[:-1], route[1:]))
        return float(value - self.lmb*travel), float(value), int(travel)

    def solve_with_must(self):
        """
        Construye la ruta insertando primero todos los must,
        y luego intenta insertar greedy el resto de eventos
        siempre que aumenten el objetivo.
        """
        route = [0, self.N + 1]  # start → end
        sched = {}
        stats = {"objective": 0.0, "value": 0.0, "travel": 0}

        def try_insert(route, k, pos):
            new_route = route[:pos+1] + [k] + route[pos+1:]
            feasible, new_sched = self._schedule(new_route)
            if feasible:
                obj, val, travel = self._route_value(new_route)
                return True, new_route, new_sched, obj, val, travel
            return False, None, None, None, None, None

        # 🔹 1. Insertar todos los must primero (aunque bajen el objetivo)
        for k in sorted(self.must):
            best_gain, best_sched, best_route = None, None, None
            for pos in range(len(route)-1):
                ok, new_route, new_sched, obj, val, travel = try_insert(route, k, pos)
                if ok and (best_gain is None or obj > best_gain):
                    best_gain, best_sched, best_route = obj, new_sched, new_route
                    stats = {"objective": obj, "value": val, "travel": travel}
            if best_gain is not None:
                route, sched = best_route, best_sched

        # 🔹 2. Intentar insertar greedy el resto de eventos
        others = set(self.events["event_id"].astype(int)) - self.must
        improved = True
        while improved:
            improved = False
            best_gain, best_sched, best_route, best_ev = None, None, None, None
            for k in sorted(others):
                for pos in range(len(route)-1):
                    ok, new_route, new_sched, obj, val, travel = try_insert(route, k, pos)
                    if ok and (best_gain is None or obj > best_gain):
                        # Solo aceptamos si mejora el objetivo
                        if obj > stats["objective"]:
                            best_gain, best_sched, best_route, best_ev = obj, new_sched, new_route, k
                            new_stats = {"objective": obj, "value": val, "travel": travel}
            if best_gain is not None:
                route, sched, stats = best_route, best_sched, new_stats
                others.remove(best_ev)
                improved = True

        # 🔹 3. Si no se insertó nadie, al menos devolver la ruta trivial
        if not sched:
            feasible, sched = self._schedule(route)
            obj, val, travel = self._route_value(route)
            stats = {"feasible": feasible, "objective": obj, "value": val, "travel": travel}

        return route, sched, stats

# ---------------------------------------------------------------------------
# Build nodes
# ---------------------------------------------------------------------------
def load_daystart_coords(csv_path: str = DAYSTART_CSV):
    if not os.path.exists(csv_path):
        return None, None
    try:
        df = pd.read_csv(csv_path)
        if {"lat","lon"}.issubset(df.columns) and not df.empty:
            return float(df.iloc[0]["lat"]), float(df.iloc[0]["lon"])
    except Exception:
        pass
    return None, None

def build_nodes_and_matrix(events, day, start_min=0, end_min=None,
                           start_lat=None, start_lon=None):
    """
    Build nodes (start, events, end) and the travel time matrix.
    events: DataFrame con event_id, lat, lon, duration_min, earliest_start, latest_start
    """
    if end_min is None:
        end_min = to_minutes_since(day, pd.to_datetime(events["latest_start"].max()).to_pydatetime())

    # depot coords
    if start_lat is None or start_lon is None:
        dep_lat, dep_lon = load_daystart_coords()
        if dep_lat is not None and dep_lon is not None:
            start_lat, start_lon = dep_lat, dep_lon
        else:
            start_lat, start_lon = events.iloc[0]["lat"], events.iloc[0]["lon"]

    nodes = [{
        "idx": 0, "lat": float(start_lat), "lon": float(start_lon),
        "duration": 0, "a": start_min, "b": start_min, "name": "Inicio"
    }]

    for _, r in events.iterrows():
        nodes.append({
            "idx": int(r["event_id"]),
            "lat": float(r["lat"]), "lon": float(r["lon"]),
            "duration": int(r["duration_min"]),
            "a": to_minutes_since(day, pd.to_datetime(r["earliest_start"]).to_pydatetime()),
            "b": to_minutes_since(day, pd.to_datetime(r["latest_start"]).to_pydatetime()),
            "name": str(r["name"]),
        })

    nodes.append({
        "idx": len(events) + 1, "lat": float(start_lat), "lon": float(start_lon),
        "duration": 0, "a": 0, "b": int(end_min), "name": "Fin"
    })

    coords = [(n["lat"], n["lon"]) for n in nodes]
    T = build_travel_matrix(coords)
    return nodes, T

# ---------------------------------------------------------------------------
# Enriquecimiento de eventos + ejecución end-to-end
# ---------------------------------------------------------------------------
def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["earliest_start","latest_start"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c])
    if "event_id" not in out.columns:
        out = out.reset_index(drop=True)
        out["event_id"] = out.index + 1
    defaults = {
        "location_score": np.nan,
        "demo_18_29": 0.0, "demo_30_44": 0.0, "demo_45_64": 0.0, "demo_65p": 0.0,
        "reach": np.nan, "must_visit": False,
        "barrio_id": "", "barrio_valor_base": np.nan, "value": np.nan,
    }
    for k, v in defaults.items():
        if k not in out.columns:
            out[k] = v
    out["must_visit"] = out["must_visit"].astype(bool)
    return out

# ---------------------------------------------------------------------------
# Normalización de valores
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Normalización de valores (shift por mínimo)
# ---------------------------------------------------------------------------
def normalize_values(df: pd.DataFrame, col: str = "value") -> pd.DataFrame:
    """
    Ajusta los valores de la columna 'col' para que el mínimo sea 0.
    Si ya son todos >= 0, no hace nada.
    """
    df = df.copy()
    if col not in df.columns or df[col].isna().all():
        return df

    vmin = df[col].min()
    if vmin < 0:
        df[col] = df[col] + abs(vmin)

    return df


def enrich_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- asegurar columnas
    if "scale" not in df.columns:
        df["scale"] = 1.0

    # Si hay valores NaN en barrio_valor_base, poner 0
    df["barrio_valor_base"] = df["barrio_valor_base"].fillna(0)

    # --- normalización
    min_val = df["barrio_valor_base"].min()
    if min_val < 0:
        df["norm_val"] = df["barrio_valor_base"] + abs(min_val)
    else:
        df["norm_val"] = df["barrio_valor_base"]

    # --- valor ponderado final
    df["value"] = df["norm_val"] * df["scale"]

    return df




def run_greedy_optimizer(
    events_csv: str = EVENTS_CSV,
    daystart_csv: str = DAYSTART_CSV,
    lambda_travel: float = 0.5,
):
    """
    Lee eventos del CSV, enriquece si falta, toma el punto de inicio y ejecuta el optimizador.
    Devuelve (stats, route, schedule).
    """
    if not os.path.exists(events_csv):
        raise FileNotFoundError(f"No existe {events_csv}")

    events = pd.read_csv(events_csv, parse_dates=["earliest_start","latest_start"])
    if events.empty:
        raise ValueError("No hay eventos para optimizar.")
    events = _ensure_schema(events)

    if events["barrio_id"].eq("").any() or events["value"].isna().any():
        events = enrich_events(events)
        events.to_csv(events_csv, index=False)  # persistimos enriquecido

    start_lat, start_lon = load_daystart(daystart_csv)
    if start_lat is None or start_lon is None:
        start_lat, start_lon = float(events.iloc[0]["lat"]), float(events.iloc[0]["lon"])

    day = pd.to_datetime(events["earliest_start"].min()).replace(hour=0, minute=0, second=0, microsecond=0)
    end_min = to_minutes_since(day, pd.to_datetime(events["latest_start"].max()).to_pydatetime())

    nodes, T = build_nodes_and_matrix(
        events=events, day=day, start_min=0, end_min=end_min,
        start_lat=start_lat, start_lon=start_lon
    )

    opt = GreedyTWOptimizer(events, T, nodes, lambda_travel=lambda_travel, day=day)
    route, sched, stats = opt.solve_with_must()
    return stats, route, sched
