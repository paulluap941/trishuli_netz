# Trishuli — automatic boulder detection

Detecting and measuring boulders (> 3.5 m) in high-resolution satellite imagery
of the Trishuli valley, Nepal, before and after the July 2025 flood.

Groundwork: 3597 hand-checked outlines across three scenes, produced with point
prompts in SAM. The automatic OrthoSAM run reached only **precision 0.16**,
because the boulders are 7.5–11 px across — far below the validated detection
limit of 30 px (Chan, Rheinwalt & Bookhagen, Earth Surf. Dynam. 14, 391–416, 2026).

This repository trains networks on that actual object size instead.

## Data

| Scene | Date | Resolution | Corridor | Outlines |
|---|---|---|---|---|
| mar | March 2025, pre-flood | 0.327 m/px | 2.22 km² | 1195 |
| nov | November 2025, post-flood | 0.466 m/px | 2.33 km² | 1182 |
| feb | February 2026, +1 year | 0.352 m/px | 2.32 km² | 791 |

Target of the application: a new Vantor scene (0.349 m/px) after a later flood.

Source data is **not** in the repository.

## Models

| | Architecture | Pretrained weights |
|---|---|---|
| Mask R-CNN | ResNet-50 + FPN | BoulderNet (Prieur et al. 2023) |
| Cellpose-SAM | ViT-L from SAM | ImageGrains 2.0 (Mair et al.) |
| Cellpose classic | U-Net | ImageGrains 1.0 |

## Layout

```
config.py                 all paths and parameters, in one place
common/                   code that runs in EVERY environment
                          (numpy / rasterio / geopandas only, no torch)
  tiles.py                tile and label generation
  folds.py                fold assignment along the valley axis
  evaluate.py             AP, IoU matching, D50 / D84
  export.py               label raster -> gpkg
requirements/             one file per environment
01_prepare_data.ipynb
02_cellpose_sam.ipynb
03_maskrcnn.ipynb
04_cellpose_unet.ipynb
05_compare.ipynb
06_vantor.ipynb
results/                  one json per run
```

## Environments

Three separate environments, because the dependencies exclude each other:
`cellpose 2.x` and `cellpose 4.x` cannot coexist, and Detectron2 pins its own
torch build.

```bash
python3 -m venv cellpose_sam_env
source cellpose_sam_env/bin/activate
pip install -r requirements/cellpose_sam.txt
python -m ipykernel install --user --name cellpose_sam --display-name "Cellpose-SAM"
```

Then pick the matching kernel per notebook in Jupyter Lab — one Jupyter, several
kernels, no manual switching.

Detectron2 is not on the pip index:

```bash
pip install "git+https://github.com/facebookresearch/detectron2.git"
```

## Paths

`config.py` reads environment variables, otherwise falls back to per-OS defaults:

```bash
export TRISHULI_RAW=/path/to/source/scenes
export TRISHULI_DATA=/path/for/generated/data
```

## Workflow

```
Phase 0   timing test: 1 fold, 10 epochs
Phase 1   Cellpose-SAM across 4 folds
Phase 2   Mask R-CNN
Phase 3   Cellpose classic
Phase 4   ablations, test block, application to Vantor
```

Metric: `AP = TP / (TP + FN + FP)` at IoU > 0.5, plus D50 and D84 against the
hand measurements — a model can count well and still measure badly.

## References

- Prieur et al. (2023), *Automatic Characterization of Boulders on Planetary
  Surfaces*, JGR Planets. [doi:10.1029/2023JE008013](https://doi.org/10.1029/2023JE008013)
- Mair et al. (2024), *Automated detecting, segmenting and measuring of grains
  in images of fluvial sediments*, ESPL. [doi:10.1002/esp.5755](https://doi.org/10.1002/esp.5755)
- Mair et al. (2026), *ImageGrains 2.0*, Earth Surf. Dynam. 14, 527.
- Stringer et al. (2021), *Cellpose*, Nature Methods.
- Chan, Rheinwalt & Bookhagen (2026), *OrthoSAM*, Earth Surf. Dynam. 14, 391–416.
