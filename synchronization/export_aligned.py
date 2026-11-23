"""Apply VisualSync's global offsets and export the common camera timeline."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import cv2
import numpy as np


def encode(directory, output, fps, first, count):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-framerate", str(fps),
         "-start_number", str(first), "-i", str(directory / "%06d.jpg"),
         "-frames:v", str(count), "-c:v", "libx264", "-threads", "2",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
        check=True,
    )


def render_grid(views, order, reference, shifts, start, end, fps, directory, movie, caption):
    columns = 4
    rows = (len(order) + columns - 1) // columns
    for index in range(start, end):
        filename = f"{index:06d}.jpg"
        sheet = np.zeros((40 + 200 * rows, 320 * columns, 3), dtype=np.uint8)
        cv2.putText(sheet, f"{caption} frame {index:06d}  |  {index / fps:.1f}s",
                    (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        for tile, name in enumerate(order):
            image = cv2.imread(str(views[name][index - shifts[name]]))
            row, column = divmod(tile, columns)
            y, x = 40 + 200 * row, 320 * column
            sheet[y:y + 180, x:x + 320] = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
            label = name.split("_cam", 1)[-1] + (" (reference)" if name == reference else "")
            cv2.putText(sheet, label, (x + 8, y + 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        if not cv2.imwrite(str(directory / filename), sheet):
            raise RuntimeError(f"Could not write preview {filename}")
    encode(directory, movie, fps, start, end - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--offsets", type=Path, required=True, help="JSON from eval_result3.py")
    parser.add_argument("--reference", required=True, help="Reference view directory name")
    parser.add_argument("--visualize", action="store_true", help="Save aligned JPEGs and the all-camera preview")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.offsets.read_text())
    reference = args.reference
    scene = reference.split("_", 1)[0]
    views = {rgb.parent.name: sorted(rgb.glob("*.jpg")) for rgb in args.workdir.glob(f"{scene}_*/rgb")}
    components = result["scenes"].get(scene, [])
    offsets = next((group["offset_frames"] for group in components if reference in group["offset_frames"]), {})
    if reference not in views or offsets.get(reference) is None:
        parser.error(f"No estimated offsets connected to {reference}")
    missing = sorted(name for name in views if offsets.get(name) is None)
    if missing:
        parser.error(f"Views without an offset to {reference}: {', '.join(missing)}")

    relative = {name: offsets[name] - offsets[reference] for name in views}
    shifts = {name: round(offset) for name, offset in relative.items()}
    start = max(shifts.values())
    end = min(len(views[name]) + shift for name, shift in shifts.items())
    if end <= start:
        parser.error("No overlap on the reference timeline")

    order = [reference, *sorted(name for name in views if name != reference)]
    output = args.output_dir
    videos = output / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="visualsync-export-") as temporary:
        frame_root = output / "frames" if args.visualize else Path(temporary)
        for name in order:
            directory = frame_root / name
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(start, end):
                source = views[name][index - shifts[name]]
                destination = directory / f"{index:06d}.jpg"
                if args.visualize:
                    shutil.copyfile(source, destination)
                else:
                    destination.symlink_to(source.resolve())
            encode(
                directory, videos / f"{name.split('_cam', 1)[-1]}.mp4",
                result["fps"], start, end - start,
            )

        if args.visualize:
            preview = output / "vis"
            preview.mkdir(exist_ok=True)
            render_grid(
                views, order, reference, shifts, start, end, result["fps"],
                preview, output / "all_cameras.mp4", "Reference",
            )
            unsynced = Path(temporary) / "unsynced"
            unsynced.mkdir()
            render_grid(
                views, order, reference, dict.fromkeys(order, 0), 0,
                min(len(view) for view in views.values()), result["fps"],
                unsynced, output / "unsynced_all_cameras.mp4", "Unsynchronized",
            )
    (output / "manifest.json").write_text(json.dumps({
        "reference": reference,
        "fps": result["fps"],
        "first_reference_frame": start,
        "end_reference_frame_exclusive": end,
        "frame_count": end - start,
        "views": {
            name: {
                "offset_to_add_to_source_frame": relative[name],
                "applied_integer_offset": shifts[name],
                "first_source_frame": views[name][start - shifts[name]].name,
            }
            for name in order
        },
    }, indent=2) + "\n")
    print(f"Exported {len(order)} views, {end - start} frames each: {output}")


if __name__ == "__main__":
    main()
