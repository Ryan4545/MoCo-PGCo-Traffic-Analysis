from dataclasses import dataclass
from pathlib import Path

@dataclass
class Paths:
    # Project root (auto)
    ROOT: Path = Path(__file__).resolve().parent

    # Raw input files (edit these)
    CRASHES_CSV: Path = ROOT / "data" / "raw" / "crashes.csv"
    CAMERAS_CSV: Path = ROOT / "data" / "raw" / "cameras.csv"
    ROADS_FILE: Path = ROOT / "data" / "raw" / "roads.geojson"      # or .shp
    SCHOOLZONES_FILE: Path = ROOT / "data" / "raw" / "schoolzones.geojson"  # optional
    LANDUSE_FILE: Path = ROOT / "data" / "raw" / "landuse.geojson"  # optional
    CITATIONS_CSV: Path = ROOT / "data" / "raw" / "citations.csv"   # optional

    # Output folders (auto)
    OUT_DIR: Path = ROOT / "data" / "processed"
    FIG_DIR: Path = ROOT / "figures"
    MODEL_DIR: Path = ROOT / "models"

@dataclass
class Columns:
    # --- CRASHES ---
    CRASH_DATE: str = "crash_date"         # parseable date/time
    CRASH_LAT: str = "latitude"
    CRASH_LON: str = "longitude"
    CRASH_GEOM_WKT: str = ""               # optional if you have WKT geometry instead of lat/lon
    CRASH_INJURY_FLAG: str = "injury"      # 1/0 or True/False
    # If you have severity categories instead, set CRASH_INJURY_FLAG empty and define:
    CRASH_SEVERITY: str = ""               # e.g., "severity"
    INJURY_SEVERITY_VALUES = {"Minor Injury", "Major Injury", "Fatal"}

    # --- CAMERAS ---
    CAMERA_ID: str = "site_id"
    CAMERA_TYPE: str = "camera_type"       # speed, red light, etc. (optional)
    CAMERA_ACTIVATION_DATE: str = "activation_date"
    CAMERA_LAT: str = "latitude"
    CAMERA_LON: str = "longitude"
    CAMERA_GEOM_WKT: str = ""              # optional if WKT geometry is provided

    # --- ROADS (optional but strongly recommended) ---
    ROAD_CLASS: str = "road_class"         # arterial/collector/local
    ROAD_NAME: str = "name"                # optional

    # --- SCHOOL ZONES (optional) ---
    SCHOOLZONE_NAME: str = "name"          # optional

    # --- LAND USE (optional) ---
    LANDUSE_TYPE: str = "landuse"          # optional

    # --- CITATIONS (optional) ---
    CITATION_CAMERA_ID: str = "site_id"
    CITATION_DATE: str = "citation_date"   # optional (monthly/daily)
    CITATION_COUNT: str = "citations"      # numeric

@dataclass
class Settings:
    # Spatial buffer around cameras to link crashes (in meters)
    CAMERA_BUFFER_METERS: float = 100.0

    # Pre/Post windows relative to activation date
    PRE_WINDOW_DAYS: int = 365
    POST_WINDOW_DAYS: int = 365

    # Target threshold for “success”
    SUCCESS_DROP_THRESHOLD: float = 0.20   # >20% injury-crash drop

    # Minimum pre injury crashes to avoid divide-by-zero / unstable rates
    MIN_PRE_INJURY_CRASHES: int = 3

    # Clustering
    N_CLUSTERS: int = 4

    # Network spillover adjacency threshold between camera sites (meters)
    NETWORK_EDGE_DISTANCE_METERS: float = 1000.0

    # Random seed
    RANDOM_STATE: int = 42


PATHS = Paths()
COLS = Columns()
SETTINGS = Settings()
