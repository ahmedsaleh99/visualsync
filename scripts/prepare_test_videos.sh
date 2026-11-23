#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python preprocess/prepare_videos.py \
    --input-dir test_vidoes/unsynced/20200305-130502_s-2700s_d-60s \
    --output-dir data/20200305-130502_s-2700s_d-60s_1fps \
    --scene capture20200305-130502 --sample-fps 1 --resume --deva-compat "$@"
