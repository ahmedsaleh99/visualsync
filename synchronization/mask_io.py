"""Read palette-indexed or DEVA RGB-encoded instance IDs."""

import numpy as np
from PIL import Image


def read_mask_ids(filename):
    with Image.open(filename) as mask:
        if mask.mode in {"RGB", "RGBA"}:
            colors = np.asarray(mask, dtype=np.int32)
            return colors[:, :, 0] + 256 * colors[:, :, 1] + 65536 * colors[:, :, 2]
        return np.array(mask)
