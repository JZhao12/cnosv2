# CNOSv2: CAD-based Novel Object Segmentation with SAM2 + DINOv3

This repository is a fork of [CNOS](https://github.com/nv-nguyen/cnos) (ICCV 2023) with two of its
models replaced:

| Stage | Original CNOS | This repository |
|---|---|---|
| Mask proposals | Segment Anything (ViT-H) | **SAM 2.1 Hiera-L** |
| Descriptors | DINOv2 ViT-L/14 | **DINOv3 ViT-L/16** (LVD-1689M weights) |
| Scope | BOP benchmark + custom objects | Custom inference only: one CAD model + one RGB image |

BOP dataset evaluation, FastSAM and detectron2 visualisation have been removed. The repository
contains only what is needed to run the SAM2 + DINOv3 pipeline.

## Status

This project is an updated version of CNOS in which the two backbone models are swapped for newer
ones, but **no quantitative benchmark results have been produced**. In particular, it has not been
evaluated on dataset benchmarks such as the BOP Challenge Datasets, so no AP / AR numbers are
reported here and it should not be assumed to reproduce the scores published for the original CNOS.

The code has only been validated qualitatively: on the example CAD model and image shipped in
`media/`, and on a handful of custom objects, the pipeline renders templates, segments the object
and returns a mask. Anyone interested in measuring it properly is welcome to run it on real
benchmark data and share the results or report issues.

## Pipeline

```
CAD model ──pyrender──▶ 42 template views ──DINOv3──▶ template descriptors
                                                          │
RGB image ──SAM2 auto-masks──▶ N proposals ──DINOv3──▶ proposal descriptors
                                                          │
                            cosine similarity (mean of top 5) ──▶ threshold ──▶ mask / vis / json
```

## Layout

```
configs/
  inference_custom.yaml            # entry point: composes `user` and `model`
  user/default.yaml                # paths to sam2_repo / dinov3_repo
  model/cnos.yaml                  # stage composition and post-processing
  model/segmentor_model/sam2.yaml  # SAM2 config, checkpoint, proposal thresholds
  model/descriptor_model/dinov3.yaml
src/
  model/sam2.py                    # SAM2 automatic mask generator, CNOS wrapper
  model/dinov3.py                  # DINOv3 descriptor wrapper
  model/detector.py                # two-stage container and template matching
  model/utils.py, model/loss.py    # Detections structure, similarity
  poses/                           # template viewpoints and pyrender rendering
  scripts/render_custom.sh         # step 1: render templates
  scripts/run_inference_custom.sh  # step 2: run inference
  scripts/inference_custom.py
  utils/                           # bbox / IO / trimesh helpers
media/teabox/                      # example CAD model and RGB image
sam2_repo/                         # SAM2 upstream repo (cloned during setup, not tracked)
dinov3_repo/                       # DINOv3 upstream repo + weights/ (cloned during setup, not tracked)
```

## Installation

The environment is managed with mamba, named `cnosv2`, Python 3.10, PyTorch 2.8 with CUDA 12.8.

```bash
mamba create -n cnosv2 python=3.10
mamba activate cnosv2
pip install -r requirements.txt
```

Alternatively use `mamba env create -f environment.yml` from the repository root; it reads
`requirements.txt` next to it.

### SAM2

```bash
git clone https://github.com/facebookresearch/sam2.git sam2_repo
pip install -e sam2_repo
wget -P sam2_repo/checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

> Clone into `sam2_repo`, never into a directory named `sam2/` at the repository root: that would
> shadow the pip-installed `sam2` package.

### DINOv3

```bash
git clone https://github.com/facebookresearch/dinov3.git dinov3_repo
pip install -r dinov3_repo/requirements.txt
mkdir -p dinov3_repo/weights
```

DINOv3 weights must be requested from Meta as described in `dinov3_repo/README.md`; place
`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` in `dinov3_repo/weights/`.

Both repository paths are set in `configs/user/default.yaml`, relative to the repository root:

```yaml
dinov3_repo: ./dinov3_repo
sam2_repo: ./sam2_repo
```

## Usage

Run every command from the repository root with the `cnosv2` environment active.

```bash
export CAD_PATH=./media/teabox/mesh.obj
export RGB_PATH=./media/teabox/000000.png
export OUTPUT_DIR=./tmp/teabox
# Optional: where the mask is written, defaults to $OUTPUT_DIR/cnos_results/mask.png
# export MASK_PATH=./tmp/teabox/cnos_results/mask.png

bash ./src/scripts/render_custom.sh         # 1. render templates into $OUTPUT_DIR/*.png
bash ./src/scripts/run_inference_custom.sh  # 2. run inference, results in $OUTPUT_DIR/cnos_results/
```

Output files:

| File | Content |
|---|---|
| `cnos_results/templates.png` | Grid of templates fed to DINOv3 |
| `cnos_results/detection.npz` | Detections (boxes, scores, segmentation, ...) |
| `cnos_results/detection.json` | BOP-style flat list, `segmentation` as uncompressed RLE |
| `cnos_results/vis.png` | Input image next to the detection overlay |
| `cnos_results/mask.png` | Merged binary mask of the selected detections (1-bit) |

Calling the Python entry point exposes more options:

```bash
python -m src.scripts.inference_custom \
  --template_dir ./tmp/teabox --rgb_path ./media/teabox/000000.png \
  --num_max_dets 3 --confg_threshold 0.3 --stability_score_thresh 0.8
```

| Option | Default | Meaning |
|---|---|---|
| `--num_max_dets` | 1 | Maximum number of detections to keep |
| `--confg_threshold` | 0.5 | Similarity threshold (the spelling is inherited from upstream) |
| `--stability_score_thresh` | 0.97 | SAM2 proposal stability; the shell script passes 0.8, lower it to 0.5 for small objects |

Rendering parameters live in `src/scripts/render_custom.sh`: `LIGHTING_ITENSITY` (light intensity)
and `RADIUS` (camera-to-object distance).

## Configuration

- `configs/model/segmentor_model/sam2.yaml`: `model_cfg` / `checkpoint_path` select the SAM2 size.
  Switching to `sam2.1_hiera_s.yaml` + `sam2.1_hiera_small.pt` cuts memory use substantially.
  `points_per_batch`, `box_nms_thresh` and `min_mask_region_area` map to the automatic mask generator.
- `configs/model/descriptor_model/dinov3.yaml`: `model_name` and `weights` select the DINOv3 size;
  `image_size: 224` and `patch_size: 16` must stay divisible.
- `configs/model/cnos.yaml`: `segmentor_width_size` is the width SAM2 runs at; `post_processing_config`
  filters small boxes and masks and applies NMS; `matching_config` defines score aggregation.

## Troubleshooting

- `python: command not found`: the `cnosv2` environment is not active.
- Rendering fails with an EGL error: `src/poses/pyrender.py` uses EGL offscreen rendering, which needs
  a working GPU driver; without one, switch `PYOPENGL_PLATFORM` to `osmesa`.
- Memory: SAM2-L with DINOv3-L takes about 12 s end to end and roughly 4 GB of VRAM on a 12 GB card.
- `No template (*.png) found` during inference: run `render_custom.sh` first, or check `OUTPUT_DIR`.

## Acknowledgements

The code is based on [CNOS](https://github.com/nv-nguyen/cnos); the models come from
[SAM 2](https://github.com/facebookresearch/sam2) and [DINOv3](https://github.com/facebookresearch/dinov3).
Their licenses apply (see `LICENSE` in this repository; DINOv3 weights are governed by the DINOv3
License).

```latex
@inproceedings{nguyen2023cnos,
title={CNOS: A Strong Baseline for CAD-based Novel Object Segmentation},
author={Nguyen, Van Nguyen and Groueix, Thibault and Ponimatkin, Georgy and Lepetit, Vincent and Hodan, Tomas},
booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
pages={2134--2140},
year={2023}
}
```
