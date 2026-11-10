# CNOSv2: CAD-based Novel Object Segmentation with SAM2 + DINOv3

This repository is a fork of [CNOS](https://github.com/nv-nguyen/cnos) (ICCV 2023) with two of its
models replaced:

| Stage | Original CNOS | This repository |
|---|---|---|
| Mask proposals | Segment Anything (ViT-H) | **SAM 2.1 Hiera-L** |
| Descriptors | DINOv2 ViT-L/14 | **DINOv3 ViT-L/16** (LVD-1689M weights) |
| Scope | BOP benchmark + custom objects | BOP benchmark (pyrender or BlenderProc templates) + custom objects |

FastSAM and the detectron2 visualisation have been removed; see
[Testing on BOP datasets](#testing-on-bop-datasets) for what is supported.

## Status

The two backbone models are swapped for newer ones. The custom single-image pipeline has been
validated qualitatively on the CAD models and images shipped in `media/`.

BOP evaluation is ported (`run_inference.py` + local `bop_toolkit` scoring) and has been run on the
seven BOP'23 core datasets with both kinds of templates of the paper. With pyrender templates the mean
2D segmentation AP over the seven datasets is **45.5**, against 36.1 (SAM + pyrender) and 40.4
(SAM + BlenderProc) reported for the original CNOS. With BlenderProc templates (frames of the
BlenderProc4BOP training split) the mean over the five locally scored datasets is 45.2, against 45.5
for pyrender on the same five: unlike in the paper, realistic templates bring no gain on top of
SAM2 + DINOv3; see [Results](#results). ITODD and HB were scored on the BOP website, the other
five locally.

## Pipeline

```
CAD model ──pyrender──▶ 42 template views ──DINOv3──▶ template descriptors
                                                          │
RGB image ──SAM2 auto-masks──▶ N proposals ──DINOv3──▶ proposal descriptors
                                                          │
                            cosine similarity (mean of top 5) ──▶ threshold ──▶ mask / vis / json
```

## Method description

Text used for the method entry on the [BOP website](https://bop.felk.cvut.cz/) (its description template).

**Submitted to:** BOP Challenge 2025, model-based 2D detection and 2D segmentation of unseen objects,
BOP-Classic-Core (LM-O, T-LESS, TUD-L, IC-BIN, ITODD, HB, YCB-V), BOP'19 test targets.

**Training data:** none. CNOSv2 is zero-shot: SAM 2.1 Hiera-L and DINOv3 ViT-L/16 are used with their
public pretrained weights (`sam2.1_hiera_large.pt`, `dinov3_vitl16_pretrain_lvd1689m`), no fine-tuning
on BOP data.

**Onboarding data:** 42 templates per object rendered with pyrender from the CAD model (viewpoints of
the level-0 icosphere, camera distance 0.4 of the predefined poses, black background, uniform grey
material for T-LESS). No real onboarding images.

**Used 3D models:** the default BOP CAD models (`models/`, `models_cad/` for T-LESS).

**Notes:** the three-stage CNOS pipeline (Nguyen et al., ICCV-W 2023) with newer backbones.
(1) Proposals: SAM 2.1 automatic mask generator on the RGB image resized to 640 px width,
`points_per_side` 32, `pred_iou_thresh` 0.8, `stability_score_thresh` 0.85 (the SAM v1 value 0.97 is too
strict for SAM2 and discards most proposals), `box_nms_thresh` 0.7; proposals smaller than 5% of the
image side or 0.03% of the image area are dropped. (2) Descriptors: each proposal is masked, cropped,
padded to a square and resized to 224x224; the DINOv3 CLS token (1024-d) is the descriptor, the same
for the templates. (3) Matching: cosine similarity between a proposal and the 42 templates of every
object, aggregated as the mean of the top 5; the object with the highest score is assigned, proposals
below 0.15 are rejected, per-object NMS at IoU 0.25, at most 100 instances per image. The same
prediction file serves both tasks (modal mask as RLE for segmentation, amodal box for detection);
`time` is the wall time of stages 1 to 3 per image. Inference runs in `bf16-mixed` with
pytorch-lightning. Code: https://github.com/JZhao12/cnosv2.

### Computer specifications

| Component | Value |
|---|---|
| GPU | 1x NVIDIA GeForce RTX 5070, 12 GB |
| CPU | Intel Core Ultra 9 285K, 24 cores |
| RAM | 62 GB |
| Storage | Samsung 990 EVO Plus NVMe SSD, 2 TB |
| OS | Ubuntu 24.04.5 LTS, Linux 7.0.0-42-generic |
| Driver / CUDA | NVIDIA driver 595.99.02, CUDA 12.8, cuDNN 9.10 |
| Software | Python 3.10.22, PyTorch 2.8.0+cu128, pytorch-lightning 2.5.4, SAM 2.1, DINOv3 (torch.hub, local), bop_toolkit |
| Runtime | about 1.2 s per image end to end (SAM2 proposals ~1.1 s, DINOv3 descriptors + matching ~0.1 s); template rendering (642 views per object) under a minute per dataset; GPU memory 5 to 10 GB depending on image size (640x480 to 1280x960) |

## Layout

```
configs/
  inference_custom.yaml            # custom inference entry point: composes `user` and `model`
  run_inference.yaml               # BOP inference entry point (+ machine, data)
  render.yaml                      # BOP template rendering
  user/default.yaml                # paths to sam2_repo / dinov3_repo / BOP root dir
  machine/local.yaml, machine/trainer/local.yaml  # pytorch-lightning 2.x Trainer (bf16-mixed)
  data/bop.yaml                    # BOP datasets table, query / template dataloaders
  model/cnos.yaml                  # stage composition, post-processing, onboarding (templates)
  model/segmentor_model/sam2.yaml  # SAM2 config, checkpoint, proposal thresholds
  model/descriptor_model/dinov3.yaml
run_inference.py                   # BOP: SAM2 + DINOv3 on a test split -> BOP-format json
src/
  dataloader/bop.py                # BOP test images (test_targets) and pyrender templates
  model/sam2.py                    # SAM2 automatic mask generator, CNOS wrapper
  model/dinov3.py                  # DINOv3 descriptor wrapper
  model/detector.py                # two-stage container, template matching, BOP test_step
  model/utils.py, model/loss.py    # Detections structure, similarity
  poses/                           # template viewpoints and pyrender rendering
  scripts/render_custom.sh         # custom step 1: render templates
  scripts/run_inference_custom.sh  # custom step 2: run inference
  scripts/inference_custom.py
  scripts/render_template_with_pyrender.py  # BOP: render templates of a whole dataset
  scripts/download_bop_classic.sh  # BOP: download + unzip the BOP'23 core datasets
  scripts/run_bop_dataset.sh       # BOP: templates -> inference -> scores for one dataset (smoke|full)
  scripts/eval_bop_coco.sh         # BOP: COCO AP/AR with bop_toolkit, locally
  scripts/summarize_bop_results.py # BOP: results table aligned with Table 1 of the paper
  scripts/ablate_sam2_ycbv.py      # SAM2 proposal-parameter ablation on a YCB-V subset
  utils/                           # bbox / IO / trimesh helpers
media/teabox/                      # example CAD model and RGB image
datasets/bop23_challenge/          # BOP datasets, templates and results (not tracked)
sam2_repo/                         # SAM2 upstream repo (cloned during setup, not tracked)
dinov3_repo/                       # DINOv3 upstream repo + weights/ (cloned during setup, not tracked)
bop_toolkit_repo/                  # bop_toolkit (cloned during setup, not tracked)
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
local_root_dir: ./datasets/bop23_challenge   # BOP datasets / templates / results
```

### bop_toolkit (BOP evaluation only)

`bop_toolkit` pins `numpy<2`, which would downgrade numpy and break `opencv-python` 5 in this
environment, so it is installed without its dependencies (only `pypng` is actually missing). The
BOP evaluation uses a `pycocotools` fork that keeps the `ignore` flag of the ground truth:

```bash
git clone --depth 1 https://github.com/thodan/bop_toolkit.git bop_toolkit_repo
pip install --no-deps -e ./bop_toolkit_repo && pip install pypng
pip install --no-deps "git+https://github.com/MartinSmeyer/cocoapi.git#subdirectory=PythonAPI"
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
| `--stability_score_thresh` | 0.85 | SAM2 proposal stability; the shell script passes 0.8, lower it to 0.5 for small objects. The SAM v1 value 0.97 drops most SAM2 proposals |

Rendering parameters live in `src/scripts/render_custom.sh`: `LIGHTING_ITENSITY` (light intensity)
and `RADIUS` (camera-to-object distance).

## Testing on BOP datasets

Same layout as the original CNOS: everything lives under `configs/user/default.yaml:local_root_dir`
(`./datasets/bop23_challenge`).

### Quick path: one dataset end to end

```bash
bash src/scripts/download_bop_classic.sh lmo          # no argument = all seven BOP'23 core datasets
bash src/scripts/run_bop_dataset.sh lmo smoke         # 5 images: checks templates -> inference -> scoring
bash src/scripts/run_bop_dataset.sh lmo full          # all BOP'19 test targets
bash src/scripts/download_train_pbr.sh lmo            # BlenderProc4BOP frames (22 GB), then:
RENDERING=pbr bash src/scripts/run_bop_dataset.sh lmo full
python -m src.scripts.summarize_bop_results           # table aligned with Table 1 of the paper
```

`run_bop_dataset.sh` renders the templates if they are missing, runs the inference, validates the
prediction file with `bop_toolkit` and scores it locally. Concurrent invocations serialise their GPU
stages through a file lock (`GPU_LOCK`, default `/tmp/cnosv2_gpu.lock`). The individual steps are
described below.

### 1. Dataset

`src/scripts/download_bop_classic.sh` fetches `<ds>_base.zip`, `<ds>_models.zip` and the BOP'19 test
archive from the Hugging Face `bop-benchmark` mirror (resumable, archives kept in `datasets/tmp/`)
and unzips them so that the folder looks like:

```
datasets/bop23_challenge/datasets/ycbv/
  test/000048/{rgb,depth,mask,mask_visib,scene_gt.json,scene_gt_info.json,scene_camera.json}   # ycbv_test_bop19.zip
  models/obj_000001.ply ...                                                                        # ycbv_models.zip
  test_targets_bop19.json, camera_*.json                                                           # ycbv_base.zip
```

Only the images listed in `test_targets_bop19.json` are processed (900 for YCB-V). Dataset
specifics are handled automatically: `test_primesense` split for T-LESS and HB, `models_cad` for
T-LESS, 8-bit grayscale `gray/*.tif` for ITODD, and the non-contiguous object ids of LM-O
(`[1, 5, 6, 8, 9, 10, 11, 12]`), which are taken from the template folder names.

### 2. Templates

Two kinds of templates, selected with `model.onboarding_config.rendering_type` (rows 6 and 7 of
Table 1 of the paper):

**pyrender** (default):

```bash
python -m src.scripts.render_template_with_pyrender dataset_name=ycbv   # `all` = seven BOP'23 core datasets
```

Renders the 642 views of the dense sphere for every CAD model into
`datasets/bop23_challenge/datasets/templates_pyrender/ycbv/obj_XXXXXX/` (about 50 s and 470 MB for
YCB-V). The number of views actually used as references is `model.onboarding_config.level_templates`
(0: 42 views, 1: 162, 2: 642); their DINOv3 descriptors are cached next to the templates.

**pbr** (BlenderProc): no rendering; the templates are real frames of the photo-realistic
[BlenderProc4BOP](https://bop.felk.cvut.cz/datasets/) training split of the dataset.

```bash
bash src/scripts/download_train_pbr.sh ycbv   # no argument = all seven; 21-37 GB per dataset
```

Downloads `<dataset>_train_pbr.zip` from the Hugging Face mirror (8 parallel byte-range streams,
resumable) and extracts it to `datasets/bop23_challenge/datasets/<dataset>/train_pbr/` (LM-O uses the
LM archive). At inference time `src/dataloader/bop_pbr.py` picks, for each of the 42 template
viewpoints and each object, the instance of the first 10 scenes (10 x 1000 frames) whose viewing
direction is the closest among the instances that are more than 80% visible; the template is the
frame with the background blacked out by `mask_visib` and cropped to the mask, exactly like a
proposal. The selection is written to `<dataset>/train_pbr_templates_level0.json` and can be
inspected with `python -m src.dataloader.bop_pbr ycbv` (one image grid per object in
`logs/pbr_templates/`). One difference with the original implementation: the nearest instance is
searched among all candidates, while the original drew 5000 of them at random (unseeded, with
replacement) before the search, so its templates were not reproducible and not always the closest.

### 3. Inference

```bash
python run_inference.py dataset_name=ycbv
# denser templates
python run_inference.py dataset_name=ycbv model.onboarding_config.level_templates=1
# smoke test on 5 images
python run_inference.py dataset_name=ycbv machine.trainer.limit_test_batches=5
# BlenderProc templates (after download_train_pbr.sh ycbv)
python run_inference.py dataset_name=ycbv model.onboarding_config.rendering_type=pbr
```

Writes one `.npz` per image and the merged BOP-format predictions (COCO results with uncompressed
RLE masks) to
`datasets/bop23_challenge/results/cnos_exps/CustomSam2AutomaticMaskGenerator_template_pyrender0_aggavg_5_ycbv.json`
(`_template_pbr0_` with BlenderProc templates). That file can be uploaded to the
[BOP website](https://bop.felk.cvut.cz/) or scored locally:

### 4. Evaluation

```bash
DATASET_NAME=ycbv bash src/scripts/eval_bop_coco.sh \
  datasets/bop23_challenge/results/cnos_exps/CustomSam2AutomaticMaskGenerator_template_pyrender0_aggavg_5_ycbv.json
# BlenderProc templates: the method name has to differ (bop_toolkit parses METHOD_DATASET-SPLIT.json)
DATASET_NAME=ycbv bash src/scripts/eval_bop_coco.sh \
  datasets/bop23_challenge/results/cnos_exps/CustomSam2AutomaticMaskGenerator_template_pbr0_aggavg_5_ycbv.json cnosv2pbr
```

The script computes `scene_gt_coco.json` once (`bop_toolkit/scripts/calc_gt_coco.py`), then runs
`eval_bop22_coco.py` for `segm` (BOP 2D segmentation task) and amodal `bbox` (2D detection task).
Scores land in `datasets/bop23_challenge/results/bop_eval/cnosv2_ycbv-test/scores_bop22_coco_*.json`.

The BOP'19 test archives of ITODD and HB ship no `scene_gt.json`, so these two cannot be scored
locally: `run_bop_dataset.sh` stops after writing the prediction file, which has to be uploaded to
the [BOP website](https://bop.felk.cvut.cz/) for scoring.

### Results

All numbers: BOP'19 test targets, SAM 2.1 Hiera-L proposals (`stability_score_thresh` 0.85), DINOv3
ViT-L/16 descriptors, 42 templates per object (level 0) either rendered with pyrender or picked from
the BlenderProc4BOP frames (see [2. Templates](#2-templates)), `bf16-mixed`, single RTX 5070
(see [Method description](#method-description)). LM-O, T-LESS, TUD-L, IC-BIN and YCB-V are scored
locally with `bop_toolkit`; ITODD and HB by the BOP website. Tables are generated with
`python -m src.scripts.summarize_bop_results`.

#### 2D segmentation AP (BOP'19 test targets, 42 template views)

| Method | LM-O | T-LESS | TUD-L | IC-BIN | ITODD | HB | YCB-V | Mean |
|---|---|---|---|---|---|---|---|---|
| CNOS (SAM + DINOv2 + Pyrender) | 33.3 | 38.3 | 35.8 | 27.2 | 14.8 | 45.9 | 57.6 | 36.1 |
| CNOS (SAM + DINOv2 + BlenderProc) | 39.6 | 39.7 | 39.1 | 28.4 | 28.2 | 48.0 | 59.5 | 40.4 |
| CNOS (FastSAM + DINOv2 + BlenderProc) | 39.7 | 37.4 | 48.0 | 27.0 | 25.4 | 51.1 | 59.9 | 41.2 |
| CNOSv2 (SAM2 + DINOv3 + Pyrender) | 41.7 | 46.3 | 46.6 | 38.6 | 23.7 | 56.3 | 65.6 | 45.5 |
| CNOSv2 (SAM2 + DINOv3 + BlenderProc) | 42.9 | 43.3 | 37.9 | 36.3 | - | - | 65.5 | 45.2 |

The first 3 rows are Table 1 of the CNOS paper (ICCV-W 2023). The BlenderProc row of CNOSv2 is
awaiting the BOP website scores of ITODD and HB (prediction files in
`datasets/bop23_challenge/results/bop_submission_pbr/`); its mean is over the five scored datasets,
where the pyrender row averages 47.8.

#### 2D detection AP (amodal bbox), ours only

| Method | LM-O | T-LESS | TUD-L | IC-BIN | ITODD | HB | YCB-V | Mean |
|---|---|---|---|---|---|---|---|---|
| CNOSv2 (SAM2 + DINOv3 + Pyrender) | 41.6 | 36.5 | 40.0 | 26.0 | - | - | 57.3 | 40.3 |
| CNOSv2 (SAM2 + DINOv3 + BlenderProc) | 42.3 | 31.5 | 33.8 | 24.3 | - | - | 51.0 | 36.6 |

#### Per-dataset details: pyrender templates

| Dataset | images | detections | segm AP | AP50 | AP75 | AR100 | bbox AP | time / image | scored |
|---|---|---|---|---|---|---|---|---|---|
| LM-O | 200 | 10839 | 41.7 | 69.8 | 46.4 | 50.7 | 41.6 | 1.27 s | local |
| T-LESS | 1000 | 49646 | 46.3 | 57.6 | 53.1 | 55.7 | 36.5 | 1.28 s | local |
| TUD-L | 600 | 17544 | 46.6 | 75.9 | 52.6 | 52.3 | 40.0 | 1.17 s | local |
| IC-BIN | 150 | 4358 | 38.6 | 66.1 | 41.7 | 44.9 | 26.0 | 1.25 s | local |
| ITODD | 721 | 5872 | 23.7 | 35.5 | 27.4 | 37.9 | - | 1.14 s | BOP website |
| HB | 300 | 11206 | 56.3 | 70.6 | 65.6 | 59.8 | - | 1.24 s | BOP website |
| YCB-V | 900 | 23503 | 65.6 | 87.0 | 76.4 | 72.8 | 57.3 | 1.24 s | local |

ITODD, HB: the BOP'19 test archives ship no scene_gt.json, so these scores come from the BOP website (same eval_bop22_coco.py); raw score files in docs/bop_website_scores/. Their bbox AP is listed only if the 2D detection task was also scored there.

The files uploaded to the BOP website (one per dataset, `cnosv2_<dataset>-test*.json[.gz]`) are written by `run_bop_dataset.sh` and collected in `datasets/bop23_challenge/results/bop_submission/`.

#### Per-dataset details: BlenderProc templates

| Dataset | images | detections | segm AP | AP50 | AP75 | AR100 | bbox AP | time / image | scored |
|---|---|---|---|---|---|---|---|---|---|
| LM-O | 200 | 11526 | 42.9 | 72.7 | 48.2 | 51.3 | 42.3 | 1.26 s | local |
| T-LESS | 1000 | 50377 | 43.3 | 53.3 | 49.7 | 54.0 | 31.5 | 1.28 s | local |
| TUD-L | 600 | 16853 | 37.9 | 63.4 | 42.9 | 52.4 | 33.8 | 1.15 s | local |
| IC-BIN | 150 | 4633 | 36.3 | 62.8 | 39.4 | 44.3 | 24.3 | 1.21 s | local |
| ITODD | 721 | 5827 | - | - | - | - | - | 1.14 s | BOP website, pending |
| HB | 300 | 11469 | - | - | - | - | - | 1.22 s | BOP website, pending |
| YCB-V | 900 | 23704 | 65.5 | 87.0 | 76.6 | 71.2 | 51.0 | 1.18 s | local |

Prediction files for the BOP website (method name `cnosv2pbr`): `datasets/bop23_challenge/results/bop_submission_pbr/`.

**Pyrender vs BlenderProc templates.** The paper (rows 6 and 7) gained 4.3 AP from the realistic
templates with SAM + DINOv2. With SAM2 + DINOv3 the two template kinds are on par on average
(LM-O +1.2, YCB-V -0.1, IC-BIN -2.3) and the realistic ones lose on the two datasets with untextured
objects: T-LESS (-3.0) and TUD-L (-8.7). The selected templates were inspected
(`python -m src.dataloader.bop_pbr <dataset>`, viewing-direction error 0.4 to 3.2 degrees on
average, 17.6 on LM-O whose PBR scenes only show the objects from above). What differs from the
pyrender renderings: the BlenderProc4BOP frames light the untextured objects with random coloured
illumination (T-LESS parts come out green, pink or grey; the white TUD-L ape pink or blue), the
objects are small (median mask side 88 px on TUD-L, 95 px on LM-O, against 406 px for the pyrender
renderings) and up to 20% occluded (`visib_fract` > 0.8). On TUD-L the realistic templates of the
white ape then match almost every white blob: 80% of all detections are labelled as that object
(a third with pyrender templates). The 2D detection AP drops more than the segmentation AP with
BlenderProc templates (YCB-V 57.3 -> 51.0 for the same proposals): the boxes of identical masks are
identical in both runs, the realistic templates simply rank and label the SAM2 proposals differently.

#### YCB-V: effect of the SAM2 stability threshold

`stab` is `stability_score_thresh` in `configs/model/segmentor_model/sam2.yaml` (default 0.85):

| SAM2 stab | Task | AP | AP50 | AP75 | AR100 | time / image |
|---|---|---|---|---|---|---|
| 0.85 (default) | 2D segmentation | 65.6 | 87.0 | 76.4 | 72.8 | 1.24 s |
| 0.85 (default) | 2D detection (amodal bbox) | 57.3 | 81.5 | 64.3 | 66.4 | 1.24 s |
| 0.90 | 2D segmentation | 65.1 | 86.0 | 75.8 | 72.2 | 1.19 s |
| 0.90 | 2D detection (amodal bbox) | 57.5 | 80.9 | 64.5 | 66.3 | 1.19 s |
| 0.97 (SAM v1 value) | 2D segmentation | 51.7 | 63.4 | 60.9 | 55.1 | 1.02 s |
| 0.97 (SAM v1 value) | 2D detection (amodal bbox) | 45.3 | 60.1 | 50.5 | 49.4 | 1.02 s |

The stability threshold matters: 0.97 was inherited from the original SAM v1 config, but SAM2's
stability scores are distributed differently and that value discards most of its proposals
(proposal recall at IoU 0.5 drops from 98% to 71% on YCB-V; dark objects such as the clamps and the
marker get no proposal at all). DINOv3 matching is not the bottleneck: when a proposal overlaps a
ground-truth instance, its label is correct 99.9% of the time. Denser point grids
(`points_per_side=64`) and `use_m2m` did not help in a subset ablation (`python -m src.scripts.ablate_sam2_ycbv`).

### Not ported

- FastSAM, model-free onboarding (HOT3D / HANDAL videos), multi-GPU / SLURM.

## Configuration

- `configs/model/segmentor_model/sam2.yaml`: `model_cfg` / `checkpoint_path` select the SAM2 size.
  Switching to `sam2.1_hiera_s.yaml` + `sam2.1_hiera_small.pt` cuts memory use substantially.
  `points_per_batch`, `box_nms_thresh` and `min_mask_region_area` map to the automatic mask generator.
  `stability_score_thresh` (0.85) controls how many proposals SAM2 keeps; see [Results](#results).
- `configs/model/descriptor_model/dinov3.yaml`: `model_name` and `weights` select the DINOv3 size;
  `image_size: 224` and `patch_size: 16` must stay divisible.
- `configs/model/cnos.yaml`: `segmentor_width_size` is the width SAM2 runs at; `post_processing_config`
  filters small boxes and masks and applies NMS; `matching_config` defines score aggregation;
  `onboarding_config.level_templates` picks the template density for BOP runs.
- `configs/machine/trainer/local.yaml`: pytorch-lightning 2.x `Trainer` used by `run_inference.py`
  (`precision: bf16-mixed`, single GPU, `limit_test_batches` for smoke tests).

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
