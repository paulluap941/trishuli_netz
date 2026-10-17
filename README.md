# trishuli_netz

Finding and measuring boulders in sub-metre satellite imagery of the Trishuli
valley in Nepal, before and after the July 2025 flood.

## Why

I mapped 3597 boulders by hand across three scenes, using point prompts in SAM.
Running OrthoSAM fully automatically only got precision 0.16, which turned out
not to be my fault: the boulders are 7.5 to 11 px across and the published
detection limit is 30 px (Chan, Rheinwalt & Bookhagen 2026). So the tool was
being used an order of magnitude below what it was validated for.

The idea here is to fine-tune networks on the size the boulders actually are,
so the next scene does not have to be clicked through by hand.

## Data

| Scene | Date | Resolution | Corridor | Outlines |
|-------|------|-----------:|---------:|---------:|
| mar | March 2025, before the flood | 0.327 m/px | 2.22 km² | 1195 |
| nov | November 2025, after | 0.466 m/px | 2.33 km² | 1182 |
| feb | February 2026, a year later | 0.352 m/px | 2.32 km² | 791 |

The imagery is commercial and not in here. Point `TRISHULI_RAW` at it.

What this should eventually run on is a Vantor scene at 0.349 m/px covering a
later flood.

## Models

| | Architecture | Pretrained on |
|---|---|---|
| Cellpose-SAM | ViT-L from SAM | ImageGrains 2.0 |
| Mask R-CNN | ResNet-50 + FPN | BoulderNet |
| Cellpose | U-Net | ImageGrains 1.0 |

Three separate environments. `cellpose 2.x` and `cellpose 4.x` cannot be
installed together, and Detectron2 wants its own torch build.

### Cellpose-SAM

Crops come out of the archive at 384 px, cellpose takes its 256 px window from
inside them. One crop per boulder plus background crops at 30 % of the boulder
count, only at positions where the whole window is inspected ground.

| | |
|---|---|
| epochs | 150 |
| crops drawn per epoch | 250 |
| batch | 1 |
| learning rate | 1e-5 |
| scale range | 0.5, so 0.75x to 1.25x |
| contrast jitter | gain 0.8 to 1.25, offset -0.1 to 0.1 |
| normalisation | p2/p98 per window, cellpose does it |
| cellprob / flow threshold | 0.0 / 0.4 |

Rotation, flipping and scaling come from cellpose. The contrast jitter is
applied on top of them.

## Setup

```bash
python3 -m venv cellpose_sam_env
source cellpose_sam_env/bin/activate
pip install -r requirements/cellpose_sam.txt
python -m ipykernel install --user --name cellpose_sam --display-name "Cellpose-SAM"
```

Detectron2 is not on PyPI, and its `setup.py` imports torch, so pip's build
isolation has to be switched off or it fails with `No module named 'torch'`:

```bash
pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"
```

One Jupyter Lab, pick the kernel per notebook.

## Paths

```bash
export TRISHULI_RAW=/path/to/scenes
export TRISHULI_DATA=/path/for/generated/data
```

Otherwise `config.py` falls back to defaults per OS. `config.check_paths()`
complains straight away if they are wrong, which saves you from a confusing
rasterio error twenty cells later.

## Notebooks

```
01_prepare_data     build the tile archive
02_cellpose_sam     fine-tune, evaluate
03_maskrcnn
04_cellpose_unet
05_compare          read results/*.json
06_vantor           run the winner on the new scene
```

Preparation runs locally since the source scenes are there. Then copy the
archive to wherever the GPU is:

```bash
rsync -av "$TRISHULI_DATA/archive/" server:/path/to/archive/
```

## Layout

```
config.py          paths and parameters
common/            no torch in here, so it runs in all three environments
  tiles.py         tiling, label rasters, corridor masks
  folds.py         splitting along the valley
  offset.py        co-registration, only used for the corridor polygons
requirements/
results/
```

## A few things that are not obvious

Scenes are tiled in their own coordinate systems. I resample to 0.125 m/px but
do not warp anything onto a shared grid, so the labels end up exactly where I
drew them and nothing gets interpolated twice. For training the scenes do not
have to line up with each other.

Tiles carry a corridor mask. The river corridor is 26 km long and about 90 m
wide, so square tiles cover it badly, only about a third of a tile is inside it.
The rest is hillside and forest that nobody ever looked at, and treating that as
background would teach the net to suppress boulders. Pixels inside a labelled
outline are always marked valid, so an outline on the corridor edge is never
half ignored.

Folds are contiguous blocks cut at the same absolute kilometres in every scene.
Taking the last 20 % of each scene would hit different ground, since the three
reaches differ by several kilometres.

## Scoring

```
AP = TP / (TP + FN + FP)   at IoU > 0.5
```

Plus D50 and D84 against the hand measurements. A model can count boulders well
and still get their sizes wrong, and since the point of all this is a grain size
distribution, that would be the worse failure.

OrthoSAM at precision 0.16 is the bar.

## References

- Prieur et al. (2023). Automatic Characterization of Boulders on Planetary Surfaces.
  *JGR Planets*. [doi:10.1029/2023JE008013](https://doi.org/10.1029/2023JE008013)
- Mair et al. (2024). Automated detecting, segmenting and measuring of grains in
  images of fluvial sediments. *ESPL*. [doi:10.1002/esp.5755](https://doi.org/10.1002/esp.5755)
- Mair et al. (2026). ImageGrains 2.0. *Earth Surf. Dynam.* 14, 527.
- Stringer et al. (2021). Cellpose. *Nature Methods*.
- Chan, Rheinwalt & Bookhagen (2026). OrthoSAM. *Earth Surf. Dynam.* 14, 391-416.
