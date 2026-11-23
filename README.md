<br />
<p align="center">

  <h1 align="center">VisualSync: Multi-Camera Synchronization via Cross-View Object Motion</h1>

  <p align="center">
   NeurIPS, 2025
    <br />
    <a href="https://stevenlsw.github.io"><strong>Shaowei Liu*</strong></a>
    ·
    <a href="https://Davidyao99.github.io/"><strong>David Yifan Yao*</strong></a>
    ·
    <a href="https://saurabhg.web.illinois.edu/"><strong>Saurabh Gupta†</strong></a>
    ·
    <a href="https://shenlong.web.illinois.edu/"><strong>Shenlong Wang†</strong></a>
    ·
  </p>

<p align="center"> 
<img src="assets/vsync.png" alt="Demo VisualSync" />
</p>

  <p align="center">
    <a href='https://drive.google.com/file/d/1-MwQRWBm_I3576gBaC_f7D3iYcFpiox0/view?usp=sharing'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='Paper PDF'></a>
    <a href='https://arxiv.org/abs/2512.02017'><img src='https://img.shields.io/badge/arXiv-2512.02017-b31b1b.svg'  alt='Arxiv'></a>
    <a href='https://stevenlsw.github.io/visualsync/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'></a>
    <!-- <a href='' style='padding-left: 0.5rem;'><img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'></a>
    <a href='https://youtu.be/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Youtube-Video-red?style=flat&logo=youtube&logoColor=red' alt='Youtube Video'></a> -->
  </p>

</p>
<br />

This repository contains the pytorch implementation for the paper [VisualSync: Multi-Camera Synchronization via Cross-View Object Motion](https://stevenlsw.github.io/visualsync/), NeurIPS 2025. **VisualSync** aligns unsynchronized multi-view videos by matching object motion with epipolar cues. The synchronized outputs can benefit dynamic reconstruction, novel view synthesis, and multi-view data engines.


## Overview
![overview](assets/method.png)

## Update
- **2026.04.06**: Some code is not fully cleaned up yet due to graduation. Original codes are provided [here](https://drive.google.com/drive/folders/110iEHJ_xVqMq88SCua8DWnXPqw0oBBP6?usp=sharing) for reference.

## 📄 Table of Contents

- [Installation](#installation)
- [In-the-Wild Demo](#in-the-wild-demo)
- [Data Preprocessing](#preprocessing)
- [Citation](#citation)


## Installation

### Docker (NVIDIA GPU)

The Dockerfile includes PyTorch 2.5.1, CUDA 12.4 development tools, SAM2,
Grounding DINO, DEVA, CoTracker, VGGT, and MASt3R's Python dependencies.
Install Docker Engine, a compatible NVIDIA driver, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host. The host CUDA toolkit is not used. RTX 50-series / Blackwell
GPUs need a newer matching PyTorch and CUDA stack.

Use the Dev Container below for development with persistent source, data,
checkpoints, and model caches. Install the Docker Compose plugin as well.
To build the image separately from VS Code:

```bash
git submodule update --init --recursive
docker build -t visualsync:cu124 .
docker run --rm --gpus all visualsync:cu124 python -c \
  "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

CUDA extensions use the packages' built-in GPU targets and compile without a
build-time GPU. If memory is limited, add `--build-arg MAX_JOBS=1` to the build.
Dependency submodules are copied and installed before `preprocess/`, `scripts/`,
and the root README. Editing those application files only rebuilds the final
copy layers; dependency source or build-setting changes rebuild the dependency
layers. In the Dev Container, the source mount makes application edits visible
immediately without rebuilding the image.
Checkpoints are downloaded separately after starting the development container.
MASt3R's optional native RoPE extension and localization/rendering extras are
not built. The image packages this checkout; it does not add the unreleased
synchronization steps mentioned in the repository update above.

### VS Code Dev Containers

The repository includes a [Dev Container](https://code.visualstudio.com/docs/devcontainers/containers)
configuration that uses the `visualsync` service in `compose.yaml`, builds
from the Dockerfile, and mounts the checkout
at `/workspace/visualsync`. Edits in VS Code are saved directly to the host.

1. Install the **Dev Containers** extension (`ms-vscode-remote.remote-containers`).
   For a remote GPU machine, first connect using **Remote - SSH** and open this
   repository on that machine.
2. Run `git submodule update --init --recursive` on the host before opening
   the container. Docker and the NVIDIA Container Toolkit must be installed
   on the machine running the containers.
3. Build the image once on the host with `docker build -t visualsync:cu124 .`. The Dev
   Container reuses the local `visualsync:cu124` image; it does not rebuild
   it automatically when VS Code opens. Build again after changing dependencies
   or the Dockerfile.
4. Open the repository folder in VS Code, then run
   **Dev Containers: Reopen in Container** from the Command Palette.
5. Wait for post-create setup to finish. It first checks existing extensions
   and reuses them when their imports work. If needed, the setup rebuilds
   SAM2 and Grounding DINO against the mounted checkout because mounting the
   source hides the image's compiled extensions. This can take several minutes
   on first creation; ordinary reconnects reuse the container and build artifacts.

The Python interpreter is `/opt/conda/bin/python`; no `conda activate` is needed.
Use the integrated terminal to run commands directly:

```bash
bash scripts/download_weights.sh
bash scripts/prepare_test_videos.sh
python preprocess/run_dino_sam2.py \
    --workdir data/20200305-130502_s-2700s_d-60s_1fps \
    --objects person
```

Data and checkpoints persist in the mounted checkout. Model caches use the
`visualsync-model-cache` Docker volume. Compose loads credentials from the
repository's `.env` file. On a fresh checkout, run `touch .env` before
starting the container. Add `OPENAI_API_KEY` if you use GPT-generated prompts
and `HF_TOKEN` if you need authenticated Hugging Face downloads. The `.env`
file is excluded from Git and image builds. Recreate the container after
changing these values.
The development container runs as
root, so generated files in the checkout may be owned by root. If you edit
native CUDA sources, run `bash .devcontainer/post-create.sh --rebuild` to rebuild them.

### Conda

- Clone this repository: 
    ```Shell
    git clone https://github.com/stevenlsw/visualsync.git
    cd visualsync
    ```

- Install Conda Environment
    > **Note:** This installation is tested to work with **CUDA 12.4**.
    ```bash
    bash scripts/install.sh # create conda env visualsync
    conda activate visualsync
    ```

- Download Model Weights
    ```bash
    bash scripts/download_weights.sh
    ```


## In-the-Wild Demo
We show how to synchronize 3 EFL views. 

| View 1 | View 2 | View 3 |
|:-------:|:-------:|:-------:|
| ![EFL Video 1](assets/Z5TlCImQNK0_150_200.gif) | ![EFL Video 2](assets/Z5TlCImQNK0_575_625.gif) | ![EFL Video 3](assets/Z5TlCImQNK0_800_860.gif) |

The figures in this section show the EFL demo. The
[preprocessing walkthrough](#preprocessing) uses the local 11-camera
test clips.

### Step-1: Cross-view matching
---
- Visualize cross-view matching:

![cross-view-matching](assets/cross_view_matching.png)


### Step-2: Pairwise synchronization
---
- Visualize the energy landscape, the X-axis is the offsete, Y-axis is the synchronization Sampson error.
<!-- ![energy-landscape](assets/pairwise_energy.png) -->
<p align="center">
  <img src="assets/pairwise_energy.png" alt="key-frames" width="50%">
</p>

### Step-3: Global synchronization
---
- Visualize the synchronization results at each time step from all views.

![key-frames](assets/keyframe.png){width=50%}


## Preprocessing
The examples below use the local unsynchronized test clips. The 1 fps
extraction is the input for segmentation, optional DEVA tracking, and camera
pose estimation.

### 1. Dataset Preparation
---

- The dataset is organized in the following structure. The main data directory (e.g., `DATA_DIR`) can be named anything, but the subdirectories must follow this format for the preprocessing scripts to work correctly:
    ```text
    DATA_DIR/
    ├── scene1_cam1/
    │   └── rgb/
    │       ├── <img_name>1.jpg
    │       ├── <img_name>2.jpg
    │       └── ...
    ├── scene1_cam2/
    │   └── rgb/
    │       └── ...
    ├── scene1_3/
    │   └── rgb/
    │       └── ...
    └── scene2_1/
        └── rgb/
            └── ...
    ```
- **Important Formatting Rules:**
    * **Scene Grouping:** The name of each video directory must have its **scene name before the first underscore** to identify scenes and views (e.g., `scene1_cam1` and `scene1_3` are grouped as `scene1`). This is critical for the VGGT and segmentation scripts.
    * **Image Directory:** All video frames (images) must be stored in a subdirectory named `rgb`.
    * **Static Cameras:** If a video directory name contains **"cam"** (e.g., `scene1_cam1`), it is treated as a static camera. For these videos, **only the first image** will be used for pose prediction, for dynamic cameras (not include `cam` in its name) all images will be used for pose estimation.


### Prepare the provided test clips

For `test_vidoes/unsynced/20200305-130502_s-2700s_d-60s`, run in the
Dev Container terminal:

```bash
bash scripts/prepare_test_videos.sh
```

Or from the host:

```bash
docker compose run --rm -T visualsync bash scripts/prepare_test_videos.sh
```

This samples one frame per second per camera, starting at clip time zero, into
`data/20200305-130502_s-2700s_d-60s_1fps/capture20200305-130502_cam<camera>/rgb/000000.jpg`.
The common scene prefix groups all 11 views together, and `cam` identifies
these recordings as static cameras for VGGT. Each sample uses the source frame
nearest to the requested clip time. The clips retain their independent time
origins. Images wider than 1280 pixels are downscaled while preserving aspect
ratio. Use `--max-width 0` with a new output directory to preserve the original
resolution.

Each view contains `metadata.json` with source provenance, FPS, frame count,
and image dimensions. The combined manifest is saved beside the working
directory as `data/20200305-130502_s-2700s_d-60s_1fps.manifest.json`. Pass
`--visualize` to the preparation script for the camera contact sheet.
The wrapper resumes matching extractions and refuses changed source files
or settings. A partial view is decoded again to avoid reusing truncated JPEGs.
Use `python preprocess/prepare_videos.py --help` for other video folders.

This step prepares frames only. Segmentation, camera pose estimation, tracking,
and temporal synchronization are subsequent steps.

### 2. Dynamic Object Segmentation
---

#### Segment the 1 fps test clips without GPT

After preparing the test clips, run this command from the repository root
in the Dev Container:

```bash
python preprocess/run_dino_sam2.py \
    --workdir data/20200305-130502_s-2700s_d-60s_1fps \
    --objects person
```

Grounding DINO detects the requested classes in each frame, and SAM2 generates
a mask for each detection. The `person` prompt includes stationary people.
To select additional classes, pass each as a separate argument and quote
multiword names, as in `--objects person "sports ball"`.

The test run verified masks and overlays for all 660 frames across 11 cameras
(60 frames per camera). Each camera directory contains:

- `gsam2/mask/` with color-coded PNG masks and matching JSON detection records.
- `gsam2/prompt_used.json` with the prompt used for that camera.

Pass `--visualize` to save annotated JPGs under `gsam2/vis/`.

#### Optional GPT prompts and tracking

For GPT-generated class names, set `OPENAI_API_KEY` and run:

```bash
python preprocess/run_gpt.py --workdir data/20200305-130502_s-2700s_d-60s_1fps --sample 30
python preprocess/run_dino_sam2.py --workdir data/20200305-130502_s-2700s_d-60s_1fps
```

To link the per-frame masks across time with DEVA, run from the repository root.
Rerun the preparation script if you extracted the dataset earlier. It reuses
the 1 fps frames and adds the `rgb_aligned` symlinks expected by DEVA. The
symlinks point to each camera's existing frames at its original clip times.

```bash
bash scripts/prepare_test_videos.sh
(
    cd Tracking-Anything-with-DEVA
    python evaluation/eval_with_detections.py \
        --workdir ../data/20200305-130502_s-2700s_d-60s_1fps \
        --input-dir gsam2 --output-dir deva
)
```

DEVA writes its results into a `deva/` directory for each camera.

### 3. Camera Pose Estimation

Run VGGT from the repository root to estimate poses for the 11 static cameras
and save the COLMAP export:

```bash
python preprocess/vggt_to_colmap.py \
    --workdir data/20200305-130502_s-2700s_d-60s_1fps \
    --vis_path vggt_output/20200305-130502_s-2700s_d-60s_1fps --save_colmap
```

### 4. Pixel-level Tracking

Run [CoTracker3](https://github.com/facebookresearch/co-tracker) from the
repository root in the Dev Container. The offline model tracks a 10 × 10 grid
initialized on the first 1 fps frame of each camera. PyTorch downloads and
caches the pretrained weights on the first run.

```bash
python preprocess/run_cotracker.py \
    --workdir data/20200305-130502_s-2700s_d-60s_1fps
```

Each `cotracker/tracks.npz` stores the frame filenames, `(x, y)` coordinates
in extracted-image pixels, and per-point visibility. The first dimension
of both arrays follows the order of `frames`.

Pass `--visualize` to save annotated JPEGs under `cotracker/vis/` and a
tracking video at `cotracker/tracks.mp4` for each camera.
To regenerate the overlays from existing track archives, run:

```bash
python preprocess/visualize_cotracker.py \
    --workdir data/20200305-130502_s-2700s_d-60s_1fps
```

### 5. Synchronization with the author's scripts

The [author's source files](https://drive.google.com/drive/folders/110iEHJ_xVqMq88SCua8DWnXPqw0oBBP6)
are integrated under `synchronization/`. Run DEVA and VGGT first. The author’s
CoTracker runner samples points inside DEVA instance masks over several frames
and saves `tracks.pkl` under `tracks/<view>/` in the output directory.
Pass `--visualize` for per-frame JPEGs and a tracking video.

Run the pairwise pipeline from the repository root. This example uses two
overlapping Raspberry Pi views:

```bash
python synchronization/run_pair.py \
    --workdir data/20200305-130502_s-2700s_d-60s_1fps \
    --output-dir data/20200305-130502_s-2700s_d-60s_1fps_sync \
    --grid-step 16 --interval 20 \
    capture20200305-130502_camraspi04 capture20200305-130502_camraspi05
```

The pipeline runs mask-guided CoTracker3, MASt3R image matching, track
correspondence filtering, and the author’s offset search using VGGT poses.
The pair directory under `pairs/<scene>/` contains the offset result
(`result_candidates_v3_v2_vggt.pkl`). Pass `--visualize` to save the Sampson
error plot and correspondence previews. Offsets are
measured in sampled frames, so one frame is one second for this dataset.
The scorer compares frame `t` in the first view with `t + offset` in the
second. A null prediction indicates that the scorer found no suitable offset.

Run additional camera pairs into the same output directory. Use `--reuse-tracks`
when the saved tracks for those views were generated with the same inputs
and tracking settings. Then estimate
global offsets from the available pair results:

```bash
python synchronization/eval_result3.py \
    --result_roots data/20200305-130502_s-2700s_d-60s_1fps_sync/pairs \
    --use_vggt --disable_gt --dataset_fps 1 \
    --offset-output data/20200305-130502_s-2700s_d-60s_1fps_sync/offsets.json
```

The JSON lists each connected group with its own reference camera. The
author’s batch launcher, `synchronization/job_sync_v6.py`, can also rerun
offset scoring on previously generated pair correspondences. The benchmark
reporting mode of `eval_result3.py` uses ground-truth frame ranges encoded
in the benchmark view names.

#### Synchronize every Pi camera to the tripod

Run `run_pair.py` with the tripod as the first view and each Pi camera as
the second view. Use the unsynchronized 1 fps working directory as input.
Keep all pair results under the same output directory. If a direct pair
has no usable offset, score that camera against other views with reliable
offsets and let the author's global solver combine the pair estimates.

```bash
dataset=data/20200305-130502_s-2700s_d-60s_1fps
results="${dataset}_main_sync"
reference=capture20200305-130502_camtripod

for view in "$dataset"/capture20200305-130502_camraspi*; do
    python synchronization/run_pair.py \
        --workdir "$dataset" --output-dir "$results" \
        --grid-step 16 --interval 20 --offset-range 20 --reuse-tracks \
        "$reference" "${view##*/}"
done

for anchor in raspi04 raspi05; do
    for camera in raspi07 raspi11; do
        python synchronization/run_pair.py \
            --workdir "$dataset" --output-dir "$results" \
            --grid-step 16 --interval 20 --reuse-tracks \
            "capture20200305-130502_cam$anchor" "capture20200305-130502_cam$camera"
    done
done

python synchronization/eval_result3.py \
    --result_roots "$results/pairs" --use_vggt --disable_gt --dataset_fps 1 \
    --reference "$reference" --offset-output "$results/author_offsets.json"

python synchronization/export_aligned.py \
    --workdir "$dataset" --offsets "$results/author_offsets.json" \
    --reference "$reference" --output-dir "${results}/aligned"
```

The exporter uses the common time span, rounds global estimates to the 1 fps
frame grid, and saves an MP4 per camera. Pass `--visualize` to the exporter
for the combined MP4 and JPEG previews.

#### Run the full pipeline

For fixed cameras, run the author's synchronization stages from a folder
of unsynchronized source clips:

```bash
python scripts/run_pipeline.py \
    --input-dir test_vidoes/unsynced/20200305-130502_s-2700s_d-60s \
    --fps 5 --reference tripod
```

This uses Grounded SAM2, DEVA, VGGT, instance-guided CoTracker3, and MASt3R
to extract cues. It scores every camera pair with the author's discrete
Sampson-error search, then combines usable pairwise offsets with the
author's Huber IRLS solver. The reference anchors the global timeline.
The floating-point offsets are saved in `author_offsets.json` and rounded
only for the MP4 export at the requested FPS.
[VisualSync paper, Section 3](https://openreview.net/pdf/1cce1da6574f2b28e37be9c49797d96fb26b9361.pdf)

By default, the command saves the intermediate frames, masks, tracks,
correspondences, and poses used by the pipeline, plus the aligned MP4s.
Add `--visualize` to save contact sheets, per-frame overlays, score plots,
and the synchronized and unsynchronized camera previews. Use `--objects` to choose segmentation
classes (default `person`), `--interval` to choose keyframe spacing
(default 10 frames), and `--output-dir` to choose the results folder.
The default reference is the source manifest's main camera when available,
or the first camera by name. When reliable pairwise links are insufficient,
the command reports which views could not be tied to that reference.

#### Run the same workflow at 5 fps

Prepare the unsynchronized clips in a separate working directory:

```bash
clip=20200305-130502_s-2700s_d-60s
dataset="data/${clip}_5fps"
python preprocess/prepare_videos.py \
    --input-dir "test_vidoes/unsynced/$clip" --output-dir "$dataset" \
    --scene capture20200305-130502 --sample-fps 5 --max-frames 300 \
    --deva-compat --resume
```

Use that directory for SAM2, DEVA, and VGGT. Each camera has 300 input
frames. For the author's CoTracker runner, use the following options to
batch the 300-frame clips. Add `--visualize` to generate tracking previews at 5 fps:

```bash
results="${dataset}_main_sync"
for view in "$dataset"/capture20200305-130502_cam*; do
    python synchronization/run_cotracker_v5.py \
        --video_dir "$view/rgb_aligned" --mask_dir "$view/deva/Annotations" \
        --save_dir "$results/tracks/${view##*/}" \
        --grid_step 48 --interval 100 --max_query_per_batch 100 \
        --feature-chunk-size 32 --fps 5 --disable_blurry --disable_seg_filter
done
```

In the reference synchronization commands, use these saved tracks with
`--reuse-tracks`, set `--interval 100` and `--offset-range 100` for the
direct pairs, and pass `--dataset_fps 5` to the global solver. The exporter
reads the frame rate from the offsets JSON. If a fine-rate pair remains
ambiguous after testing bridge views, the author's scorer accepts
`--offset-center` and `--offset_range` to refine around a coarse VisualSync
estimate. Convert the coarse estimate into frames at the new rate first.

## Citation
```BiBTeX
@inproceedings{liu2025visualsync},
  title={VisualSync: Multi-Camera Synchronization via Cross-View Object Motion},
  author={Liu, Shaowei and Yao, David Yifan and Gupta, Saurabh and Wang, Shenlong},
  booktitle={NeurIPS},
  year={2025}
}
```


## Acknowledgement
- [Uni4D](https://github.com/Davidyao99/uni4d/) for dynamic object segmentation.
- [SAM2](https://github.com/facebookresearch/sam2) for video segmentation.
- [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA/) for video segmentation.
- [CoTracker3](https://github.com/facebookresearch/co-tracker) for video tracking. 
- [VGGT](https://github.com/facebookresearch/vggt) for camera pose estimation.
- [MASt3R](https://github.com/naver/mast3r) for cross-view correspondence.
