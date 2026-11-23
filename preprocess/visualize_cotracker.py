"""Render saved CoTracker3 point tracks over the extracted camera frames."""

import argparse
from colorsys import hsv_to_rgb
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def render_tracks(frames, frame_names, tracks, visibility, output, fps=1, max_size=None):
    """Write a numbered JPEG and a video overlay from the same rendered frames."""
    preview_dir = output.parent / "vis"
    preview_dir.mkdir(exist_ok=True)
    visibility = visibility.reshape(tracks.shape[:2])
    colors = [
        tuple(round(channel * 255) for channel in hsv_to_rgb(index * 0.618 % 1, 0.85, 1))
        for index in range(tracks.shape[1])
    ]
    with NamedTemporaryFile(dir=output.parent, suffix=".mp4", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with imageio.get_writer(temporary_path, fps=fps) as writer:
            for rgb, name, points, visible in zip(frames, frame_names, tracks, visibility, strict=True):
                frame = Image.fromarray(rgb)
                draw = ImageDraw.Draw(frame)
                for (x, y), is_visible, color in zip(points, visible, colors, strict=True):
                    if is_visible and 0 <= x < frame.width and 0 <= y < frame.height:
                        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="black")
                if max_size is not None:
                    frame.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                frame.save(preview_dir / f"{Path(name).stem}.jpg", quality=92)
                writer.append_data(np.asarray(frame))
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Frame overlays: {preview_dir}\nVideo: {output}", flush=True)


def read_frames(directory, names):
    for name in names:
        with Image.open(directory / name) as image:
            yield np.array(image.convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=1, help="Preview playback frame rate")
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")

    archives = sorted(args.workdir.glob("*/cotracker/tracks.npz"))
    if not archives:
        parser.error(f"No CoTracker3 tracks found in {args.workdir}")

    for archive in archives:
        rgb = archive.parents[1] / "rgb"
        with np.load(archive) as saved:
            frames = saved["frames"]
            tracks = saved["tracks"]
            visibility = saved["visibility"]

        render_tracks(read_frames(rgb, frames), frames, tracks, visibility, archive.with_suffix(".mp4"), args.fps)


if __name__ == "__main__":
    main()
