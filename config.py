"""Paths and parameters. Everything that I had to decide on lives here."""
import os
from pathlib import Path

WINDOWS = os.name == "nt"


def _path(variable, default):
    return Path(os.environ.get(variable, str(default))).expanduser()


# laptop vs pcpool39. env vars win if they are set.
if WINDOWS:
    _RAW = r"C:\Users\pauls\Desktop\Potsdam_Master\Internship_Uni"
    _OUT = r"C:\Users\pauls\Desktop\Potsdam_Master\Internship_Uni\trishuli_netz_data"
else:
    _RAW = "/DATA/schrepel/Internship_Uni"
    _OUT = "/DATA/schrepel/trishuli_daten"

RAW = _path("TRISHULI_RAW", _RAW)
DATA = _path("TRISHULI_DATA", _OUT)

ARCHIVE = DATA / "archive"
MODELS = DATA / "models"
RESULTS = DATA / "results"
LOGS = DATA / "logs"

# folder names are German, they come from the mapping project
OUTLINES = RAW / "ergebnisse"
MASKS = RAW / "masken_R"
VALLEY_AXIS = RAW / "talachse.gpkg"

CRS = "EPSG:32645"


# I use the panchromatic band, not the pansharpened one. Checked the
# correlation between the two over three spots per scene:
#   mar 0.90-0.95, feb 0.87-0.92, nov 0.59-0.70
# November's pansharpening clearly does something different, and since the
# net sees all three scenes it should not have to learn that. The ms files
# are only kept around to build a second luminance version for augmentation,
# because Vantor is RGB and has no pan band.
SCENES = {
    "mar": {"pan": "03Mar2025_epsg32645.tif", "ms": "03Mar2025_pan_ms.tif",
            "res": 0.3271, "date": "2025-03-03"},
    "nov": {"pan": "04Nov2025_epsg32645.tif", "ms": "04Nov2025_pan_ms.tif",
            "res": 0.4659, "date": "2025-11-04"},
    "feb": {"pan": "05Feb2026_epsg32645.tif", "ms": "05Feb2026_pan_ms.tif",
            "res": 0.3516, "date": "2026-02-05"},
}
REFERENCE = "nov"

# TODO: acquisition dates are not in the file metadata, ask about them
VANTOR = {
    "vantor_pre": {"file": "neue_daten/vantor/pre_flood_clipped.tif", "res": 0.447},
    "vantor_post": {"file": "neue_daten/vantor/post_flood_clipped2.tif", "res": 0.349},
}


# All scenes go onto one ground resolution, otherwise the 3.5 m limit is a
# different number of pixels in each (7.5 / 10.7 / 10.0 px) and the net gets
# contradictory targets. 0.125 because at 0.25 a median boulder covers only
# 4.6 cells on P2 and sits below the smallest default anchor.
RES = 0.125
TILE = 1024          # 128 m
OVERLAP = 256        # 32 m
INNER = TILE - OVERLAP

# The corridor is 26 km long and about 90 m wide, so square tiles cover it
# badly. At 1024 px roughly 35 % of a tile is inside it, at 2048 only 22 %.
# Everything outside was never looked at, so it is not a valid negative.
# Tiles that barely touch the strip are skipped.
MIN_CORRIDOR = 0.05

PERCENTILE = (2, 98)     # applied per crop on load, not baked into the tiles

MIN_DIAMETER = 3.5       # evaluation limit
MIN_TRAINING = 3.0       # kept in training, filtered out when scoring


# Cuts are absolute kilometres so the same stretch of river is training or
# validation in every scene. Doing it per scene would hit different ground,
# the reaches differ by several km (mar stops at 24.7, feb starts at 7.7).
TEST_FROM, TEST_TO = 20.9, 24.8
FOLD_EDGES = [7.5, 11.8, 15.0, 18.1, 20.9]
ALWAYS_TRAIN = [(0.0, 7.5), (24.8, 26.2)]


MODEL = {
    "cellpose_sam":  {"res": 0.125, "crop": 512,  "batch": 1},
    "maskrcnn":      {"res": 0.125, "crop": 1024, "batch": 4},
    "cellpose_unet": {"res": 0.25,  "crop": 512,  "batch": 8},
}

SERVER = "pcpool"
SERVER_DATA = "/DATA/schrepel/trishuli_daten"


def image(scene, pansharp=False):
    return RAW / SCENES[scene]["ms" if pansharp else "pan"]


def corridor(scene):
    return MASKS / f"korridor_{scene}.gpkg"


def outlines(scene):
    """outlines in november coordinates, needed for the station along the axis"""
    return OUTLINES / f"{scene}_bloecke_nov.gpkg"


def make_dirs():
    for p in (ARCHIVE, MODELS, RESULTS, LOGS):
        p.mkdir(parents=True, exist_ok=True)


def check_paths():
    # worth doing, otherwise you get a confusing rasterio error twenty cells in
    if not RAW.exists():
        raise FileNotFoundError(
            f"no source data at {RAW}\n"
            f"set TRISHULI_RAW, or change the default at the top of config.py")
    missing = [p.name for p in
               [VALLEY_AXIS, MASKS, OUTLINES] + [image(s) for s in SCENES]
               if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing under {RAW}: {', '.join(missing)}")
