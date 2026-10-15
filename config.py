"""
Central settings. All paths and parameters live here, nowhere else.

Paths resolve in this order:
  1. environment variable, if set
  2. per-OS default below (Windows = laptop, else = server)
"""
import os
from pathlib import Path

WINDOWS = os.name == "nt"


def _path(variable: str, default) -> Path:
    return Path(os.environ.get(variable, str(default))).expanduser()


if WINDOWS:
    _RAW = r"C:\Users\pauls\Desktop\Potsdam_Master\Internship_Uni"
    _OUT = r"C:\Users\pauls\Desktop\Potsdam_Master\Internship_Uni\trishuli_netz_data"
else:
    _RAW = "/DATA/schrepel/Internship_Uni"
    _OUT = "/DATA/schrepel/trishuli_daten"

RAW = _path("TRISHULI_RAW", _RAW)        # source scenes, masks, outlines
DATA = _path("TRISHULI_DATA", _OUT)      # everything we generate

ARCHIVE = DATA / "archive"               # tiles + labels
MODELS = DATA / "models"                 # checkpoints
RESULTS = DATA / "results"               # predictions, gpkg
LOGS = DATA / "logs"

OUTLINES = RAW / "ergebnisse"            # <scene>_bloecke_nov.gpkg
MASKS = RAW / "masken_R"                 # korridor_<scene>.gpkg
VALLEY_AXIS = RAW / "talachse.gpkg"

CRS = "EPSG:32645"

# --- scenes -----------------------------------------------------------
# Source is the PANCHROMATIC band, not the pansharpened one. November's
# pansharpening deviates strongly from its pan band (r = 0.59-0.70, while
# March and February reach 0.87-0.95), so pan is the more consistent basis
# across scenes. The pansharpened image is only used as a second version
# (luminance) for augmentation, because Vantor is RGB only.

SCENES = {
    "mar": {"pan": "03Mar2025_epsg32645.tif", "ms": "03Mar2025_pan_ms.tif",
            "res": 0.3271, "date": "2025-03-03"},
    "nov": {"pan": "04Nov2025_epsg32645.tif", "ms": "04Nov2025_pan_ms.tif",
            "res": 0.4659, "date": "2025-11-04"},
    "feb": {"pan": "05Feb2026_epsg32645.tif", "ms": "05Feb2026_pan_ms.tif",
            "res": 0.3516, "date": "2026-02-05"},
}
REFERENCE = "nov"                        # november's grid is the base

# target of the application, EPSG:4326, needs reprojecting
VANTOR = {
    "vantor_pre": {"file": "neue_daten/vantor/pre_flood_clipped.tif", "res": 0.447},
    "vantor_post": {"file": "neue_daten/vantor/post_flood_clipped2.tif", "res": 0.349},
}

# --- raster -----------------------------------------------------------

RES = 0.125                  # m/px, common grid for all scenes
TILE = 2048                  # px = 256 m
OVERLAP = 512                # px = 64 m
INNER = TILE - OVERLAP       # 1536 px = 192 m, only centroids in here count

PERCENTILE = (2, 98)         # per-crop normalisation, applied on load

# --- sizes ------------------------------------------------------------

MIN_DIAMETER = 3.5           # m, evaluation limit (both sides)
MIN_TRAINING = 3.0           # m, still included during training

# --- split ------------------------------------------------------------
# absolute kilometres along the valley axis, identical for all scenes.
# the two ends are always training because one scene is missing there.

TEST_FROM, TEST_TO = 20.9, 24.8
FOLD_EDGES = [7.5, 11.8, 15.0, 18.1, 20.9]     # gives 4 folds
ALWAYS_TRAIN = [(0.0, 7.5), (24.8, 26.2)]

# --- per model --------------------------------------------------------

MODEL = {
    "cellpose_sam":  {"res": 0.125, "crop": 512,  "batch": 1},
    "maskrcnn":      {"res": 0.125, "crop": 1024, "batch": 4},
    "cellpose_unet": {"res": 0.25,  "crop": 768,  "batch": 8},
}

# --- server -----------------------------------------------------------

SERVER = "pcpool"                                 # alias from ~/.ssh/config
SERVER_DATA = "/DATA/schrepel/trishuli_daten"


def image(scene: str, pansharp: bool = False) -> Path:
    """source scene, pan band by default"""
    return RAW / SCENES[scene]["ms" if pansharp else "pan"]


def corridor(scene: str) -> Path:
    return MASKS / f"korridor_{scene}.gpkg"


def outlines(scene: str) -> Path:
    """mapped boulders, in november's coordinate system"""
    return OUTLINES / f"{scene}_bloecke_nov.gpkg"


def make_dirs() -> None:
    for p in (ARCHIVE, MODELS, RESULTS, LOGS):
        p.mkdir(parents=True, exist_ok=True)
