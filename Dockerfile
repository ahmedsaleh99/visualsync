# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    PIP_CONSTRAINT=/opt/visualsync-constraints.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Keep dependency resolution from replacing the matching PyTorch/CUDA stack.
COPY <<EOF /opt/visualsync-constraints.txt
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
numpy==1.26.4
opencv-python<4.12
opencv-python-headless<4.12
transformers<5
supervision==0.22.0
EOF
RUN python -m pip install --no-cache-dir \
    'setuptools>=62.3,<75.9' wheel ninja hatchling editables

WORKDIR /workspace/visualsync
# Dependency sources get their own cache boundary. Application edits below
# must not invalidate package installation or CUDA compilation.
COPY Grounded-SAM-2/ ./Grounded-SAM-2/
COPY Tracking-Anything-with-DEVA/ ./Tracking-Anything-with-DEVA/
COPY co-tracker/ ./co-tracker/
COPY vggt/ ./vggt/
COPY mast3r/ ./mast3r/

# Fail clearly if the host checkout has uninitialized submodules.
RUN test -f Tracking-Anything-with-DEVA/pyproject.toml \
    && test -f Grounded-SAM-2/grounding_dino/setup.py \
    && test -f co-tracker/setup.py \
    && test -f vggt/pyproject.toml

ARG MAX_JOBS=2
# Grounding DINO requires this variable to enable CUDA compilation without a GPU.
# Match the architecture flags built into the vendored setup scripts.
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
RUN TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" \
    SAM2_BUILD_ALLOW_ERRORS=0 \
    python -m pip install --no-cache-dir --no-build-isolation \
    -e ./Grounded-SAM-2 -e ./Grounded-SAM-2/grounding_dino \
    -e ./Tracking-Anything-with-DEVA -e ./co-tracker -e ./vggt \
    openai 'imageio[ffmpeg]' python-dotenv 'transformers>=4.44,<5' \
    einops imgcat rich tabulate huggingface-hub safetensors matplotlib \
    -r ./mast3r/requirements.txt -r ./mast3r/dust3r/requirements.txt

# Replace the base image's Ninja wheel, whose platform metadata pip rejects.
RUN python -m pip install --no-cache-dir --upgrade ninja \
    && python -m pip check \
    && python -c "import torch, torchvision, sam2, groundingdino._C, deva, cotracker, vggt; import sam2._C; assert torch.version.cuda == '12.4'"

# Enable conda activation in interactive Bash terminals for the container user.
RUN /opt/conda/bin/conda init bash

# Frequently edited code belongs after all dependency builds and checks.
COPY preprocess/ ./preprocess/
COPY scripts/ ./scripts/
COPY README.md .gitmodules ./

CMD ["bash"]
