#!/usr/bin/env bash
set -euo pipefail
cd /workspace/visualsync

# Reuse extensions already built in the mounted checkout. Their paths match
# the image's editable installs, so reinstalls are unnecessary when imports work.
if [[ "${1:-}" != "--rebuild" ]] && python -c 'import torch, torchvision, sam2._C, groundingdino._C, deva, cotracker, vggt; assert torch.version.cuda == "12.4"'; then
    python -c 'import torch; assert torch.cuda.is_available(); print("Reusing installed extensions. GPU:", torch.cuda.get_device_name(0))'
    exit 0
fi

# The workspace mount hides native extensions and editable metadata in the image.
# Recreate them against the live checkout, using the image's installed dependencies.
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6}"
export SAM2_BUILD_ALLOW_ERRORS=0
python -m pip install --no-deps --no-build-isolation \
    -e ./Grounded-SAM-2 -e ./Grounded-SAM-2/grounding_dino \
    -e ./Tracking-Anything-with-DEVA -e ./co-tracker -e ./vggt
python -m pip check
python -c 'import torch, sam2._C, groundingdino._C; assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0))'
