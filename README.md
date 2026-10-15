# trishuli_netz

Instance segmentation of boulders in sub-metre satellite imagery of the Trishuli
valley, Nepal, before and after the July 2025 flood.

## Background

3597 boulders were mapped by hand across three scenes using point prompts in SAM.
The automatic OrthoSAM run reached precision 0.16, because the boulders are
7.5 to 11 px across while the published detection limit is 30 px
(Chan, Rheinwalt & Bookhagen 2026).

This repo fine-tunes networks on that actual object size, so the next scene does
not have to be mapped by hand.

## Data

| Scene | Date | Resolution | Corridor | Outlines |
|-------|------|-----------:|---------:|---------:|
| mar | March 2025, pre-flood | 0.327 m/px | 2.22 km² | 1195 |
| nov | November 2025, post-flood | 0.466 m/px | 2.33 km² | 1182 |
| feb | February 2026, +1 year | 0.352 m/px | 2.32 km² | 791 |

Target of the application is a Vantor scene at 0.349 m/px covering a later flood.

Source imagery is not included. Set the paths and the notebooks will find it.

## Models

| Model | Architecture | Pretrained on |
|-------|--------------|---------------|
| Cellpose-SAM | ViT-L from SAM | ImageGrains 2.0 |
| Mask R-CNN | ResNet-50 + FPN | BoulderNet |
| Cellpose | U-Net | ImageGrains 1.0 |

## Setup

Three environments, because the dependencies conflict. `cellpose 2.x` and
`cellpose 4.x` cannot coexist, and Detectron2 pins its own torch build.

```bash
python3 -m venv cellpose_sam_env
source cellpose_sam_env/bin/activate
pip install -r requirements/cellpose_sam.txt
python -m ipykernel install --user --name cellpose_sam --display-name "Cellpose-SAM"
```

Detectron2 is not on PyPI and its `setup.py` imports torch, so build isolation
has to be off:

```bash
pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"
```

Run one Jupyter Lab and pick the kernel per notebook.

## Paths

`config.py` reads two environment variables and falls back to per-OS defaults:

```bash
export TRISHULI_RAW=/path/to/source/scenes
export TRISHULI_DATA=/path/for/generated/data
```

`config.check_paths()` fails early with a readable message if they are wrong.

## Usage

```
01_prepare_data.ipynb    build the tile archive
02_cellpose_sam.ipynb    fine-tune and evaluate
03_maskrcnn.ipynb
04_cellpose_unet.ipynb
05_compare.ipynb         read results/*.json, compare
06_vantor.ipynb          apply the winner
```

Prepare the data locally, then copy the archive to wherever the GPU is:

```bash
rsync -av "$TRISHULI_DATA/archive/" server:/path/to/archive/
```

## Repository layout

```
config.py          paths and parameters, single source
common/            runs in every environment, no torch
  tiles.py         tile, label and corridor mask generation
  folds.py         fold assignment along the valley axis
  offset.py        co-registration model, used only for the corridor polygons
requirements/      one file per environment
results/           one json per run
```

## Design notes

Each scene is tiled in its own coordinate system. The images are resampled to
0.125 m/px but never warped onto a common grid, so the labels sit exactly where
they were drawn and the data is interpolated once instead of twice.

Tiles carry a corridor mask. Only about a third of a tile falls inside the mapped
strip; the rest is hillside and forest that was never inspected, so it is not a
valid negative. Pixels belonging to a labelled boulder are always marked valid,
so no outline is ever partly ignored.

Folds are contiguous blocks along the valley, cut at the same absolute kilometres
in every scene. Scene-relative splits would hit different ground in each scene,
since the three reaches differ by several kilometres.

## Metric

```
AP = TP / (TP + FN + FP)   at IoU > 0.5
```

Reported alongside D50 and D84 against the hand measurements. A model can count
well and still measure badly, which matters here because the output is a grain
size distribution.

## References

- Prieur et al. (2023). Automatic Characterization of Boulders on Planetary Surfaces.
  *JGR Planets*. [doi:10.1029/2023JE008013](https://doi.org/10.1029/2023JE008013)
- Mair et al. (2024). Automated detecting, segmenting and measuring of grains in
  images of fluvial sediments. *ESPL*. [doi:10.1002/esp.5755](https://doi.org/10.1002/esp.5755)
- Mair et al. (2026). ImageGrains 2.0. *Earth Surf. Dynam.* 14, 527.
- Stringer et al. (2021). Cellpose. *Nature Methods*.
- Chan, Rheinwalt & Bookhagen (2026). OrthoSAM. *Earth Surf. Dynam.* 14, 391-416.
