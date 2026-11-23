"""Track a grid of points in each camera's extracted frames with CoTracker3."""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image
import torch

from visualize_cotracker import render_tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True, help="Dataset containing camera directories")
    parser.add_argument("--visualize", action="store_true", help="Save per-frame overlays and a tracking video")
    parser.add_argument("--grid-size", type=int, default=10, help="Points per side of the grid")
    args = parser.parse_args()

    if args.grid_size < 1:
        parser.error("--grid-size must be positive")
    views = [(rgb, sorted(rgb.glob("*.jpg"))) for rgb in sorted(args.workdir.glob("*/rgb"))]
    if not views:
        parser.error(f"No camera rgb directories found in {args.workdir}")
    for rgb, frames in views:
        if not frames:
            parser.error(f"No JPG frames found in {rgb}")

    repository = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load(str(repository / "co-tracker"), "cotracker3_offline", source="local")
    model = model.to(device).eval()

    for rgb, frames in views:
        images = []
        for frame in frames:
            with Image.open(frame) as image:
                images.append(np.array(image.convert("RGB")))

        video = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)[None]
        with torch.inference_mode():
            tracks, visibility = model(
                video.to(device=device, dtype=torch.float32),
                grid_size=args.grid_size,
            )

        frame_names = [frame.name for frame in frames]
        tracks = tracks[0].cpu().numpy()
        visibility = visibility[0].cpu().numpy()
        output = rgb.parent / "cotracker" / "tracks.npz"
        output.parent.mkdir(exist_ok=True)
        with NamedTemporaryFile(dir=output.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with temporary_path.open("wb") as archive:
                np.savez_compressed(
                    archive,
                    frames=frame_names,
                    tracks=tracks,
                    visibility=visibility,
                )
            temporary_path.replace(output)
        finally:
            temporary_path.unlink(missing_ok=True)
        print(output, flush=True)
        if args.visualize:
            render_tracks(images, frame_names, tracks, visibility, output.with_suffix(".mp4"))


if __name__ == "__main__":
    main()
