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


# pan band, not pansharpened. correlation pan/pansharpened, three spots per
# scene: mar 0.90-0.95, feb 0.87-0.92, nov 0.59-0.70.
# ms files stay for a second luminance version, Vantor is RGB only.
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


# one ground resolution for all scenes, otherwise 3.5 m is 7.5 / 10.7 / 10.0 px
# depending on the scene. at 0.25 a median boulder covers 4.6 cells on P2 and
# sits below the smallest default anchor, at 0.125 it is 9.3 cells.
RES = 0.125
TILE = 1024          # 128 m
OVERLAP = 256        # 32 m
INNER = TILE - OVERLAP

# corridor is 26 km long and about 90 m wide, so square tiles fit it badly:
# 35 % of a 1024 px tile is inside it, 22 % of a 2048 px one. outside is
# uninspected, not background. tiles below this share are skipped.
MIN_CORRIDOR = 0.05

# ring around each outline that also counts as inspected, for boulders that
# sit on or past the corridor edge
LABEL_BUFFER = 5.0

PERCENTILE = (2, 98)     # applied per crop on load, not baked into the tiles

# dropped before tiling, so never a label and never scored
MIN_DIAMETER = 3.5


# absolute kilometres, so a stretch of river is training or validation in all
# three scenes. per scene it would hit different ground, the reaches differ by
# several km (mar stops at 24.7, feb starts at 7.7).
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


def outlines(scene, nov=False):
    """nov=False is the scene's own system, nov=True is november's"""
    return OUTLINES / (f"{scene}_bloecke_nov.gpkg" if nov else f"{scene}_bloecke.gpkg")


def make_dirs():
    for p in (ARCHIVE, MODELS, RESULTS, LOGS):
        p.mkdir(parents=True, exist_ok=True)


def check_paths():
    if not RAW.exists():
        raise FileNotFoundError(
            f"no source data at {RAW}\n"
            f"set TRISHULI_RAW, or change the default at the top of config.py")
    missing = [p.name for p in
               [VALLEY_AXIS, MASKS, OUTLINES] + [image(s) for s in SCENES]
               if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing under {RAW}: {', '.join(missing)}")
