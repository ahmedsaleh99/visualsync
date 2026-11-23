"""Run VisualSync on unsynchronized videos sampled at a common frame rate."""

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


REPOSITORY = Path(__file__).resolve().parents[1]
POSE_FILE = Path("vggt/camera_parameters.npz")


def run(script, *arguments, cwd=REPOSITORY, check=True):
    command = [sys.executable, str(cwd / script), *map(str, arguments)]
    print(f"Running {script}", flush=True)
    return subprocess.run(command, cwd=cwd, check=check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of original camera videos")
    parser.add_argument("--fps", type=float, required=True, help="Common output sampling rate")
    parser.add_argument("--output-dir", type=Path, help="Default: data/<input-folder>_<fps>fps_pipeline")
    parser.add_argument("--reference", help="Camera name or view directory. Defaults to the manifest's main camera or the first camera by name")
    parser.add_argument("--objects", nargs="+", default=["person"], help="Grounded SAM2 class prompts")
    parser.add_argument("--interval", type=int, default=10, help="Keyframe sampling interval in frames")
    parser.add_argument("--max-offset-seconds", type=float, default=20, help="Pairwise search radius in seconds")
    parser.add_argument("--visualize", action="store_true", help="Save diagnostic images, overlays, plots, and the camera grid")
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    if not math.isfinite(args.max_offset_seconds) or args.max_offset_seconds <= 0:
        parser.error("--max-offset-seconds must be finite and positive")

    if args.interval < 1:
        parser.error("--interval must be positive")

    source = args.input_dir.resolve()
    output = (args.output_dir or (
        REPOSITORY / "data" / f"{source.name}_{args.fps:g}fps_pipeline"
    )).resolve()
    workdir = output / "dataset"
    sync = output / "sync"
    visualization = ["--visualize"] if args.visualize else []
    radius = max(1, math.ceil(args.fps * args.max_offset_seconds))

    run(
        "preprocess/prepare_videos.py", "--input-dir", source, "--output-dir", workdir,
        "--scene", "capture", "--sample-fps", args.fps, "--deva-compat", "--resume", *visualization,
    )
    manifest = json.loads((output / "dataset.manifest.json").read_text())
    views = {view["view"]: view for view in manifest["views"]}
    if len(views) < 2:
        parser.error("Synchronization requires at least two camera videos")
    name = args.reference or manifest.get("reference_camera")
    if name is None:
        name = min(view["camera"] for view in views.values())
    reference = next((view for view, metadata in views.items() if name in {view, metadata["camera"]}), None)
    if reference is None:
        available = ", ".join(sorted(view["camera"] for view in views.values()))
        parser.error(f"Specify --reference using a camera name from: {available}")

    run(
        "preprocess/run_dino_sam2.py", "--workdir", workdir,
        *visualization, "--objects", *args.objects,
    )
    run(
        "evaluation/eval_with_detections.py", "--workdir", workdir,
        "--input-dir", "gsam2", "--output-dir", "deva", "--amp",
        cwd=REPOSITORY / "Tracking-Anything-with-DEVA",
    )
    colmap = ["--save_colmap"] if args.visualize else []
    run("preprocess/vggt_to_colmap.py", "--workdir", workdir, "--vis_path", output / "vggt", *colmap)

    grid_step = 48 if max(view["frame_count"] for view in views.values()) > 60 else 16
    for view, metadata in views.items():
        directory = workdir / view
        if not (directory / POSE_FILE).is_file():
            raise SystemExit(f"VGGT did not save camera parameters for {view}")
        batch_size = max(1, min(100, 30000 // metadata["frame_count"]))
        run(
            "synchronization/run_cotracker_v5.py",
            "--video_dir", directory / "rgb_aligned", "--mask_dir", directory / "deva" / "Annotations",
            "--save_dir", sync / "tracks" / view, "--fps", args.fps,
            "--grid_step", grid_step, "--interval", args.interval,
            "--max_query_per_batch", batch_size, "--feature-chunk-size", 32,
            "--disable_blurry", "--disable_seg_filter", *visualization,
        )

    ordered_views = [reference, *sorted(views.keys() - {reference})]
    # Feed the global solver only pair results produced during this invocation.
    with TemporaryDirectory(prefix="visualsync-pairs-") as temporary:
        candidates = Path(temporary)
        scene = manifest["scene"]
        (candidates / scene).mkdir()
        for first, second in combinations(ordered_views, 2):
            print(f"Scoring {first} against {second}", flush=True)
            completed = run(
                "synchronization/run_pair.py", "--workdir", workdir, "--output-dir", sync,
                "--grid-step", grid_step, "--interval", args.interval, "--offset-range", radius,
                "--fps", args.fps, "--reuse-tracks", *visualization, first, second, check=False,
            )
            pair = f"{first}__{second}"
            if completed.returncode == 3:
                (candidates / scene / pair).mkdir()
                continue
            completed.check_returncode()
            (candidates / scene / pair).symlink_to(sync / "pairs" / scene / pair, target_is_directory=True)

        offsets = output / "author_offsets.json"
        run(
            "synchronization/eval_result3.py", "--result_roots", candidates, "--use_vggt",
            "--disable_gt", "--est_choice", "irls", "--dataset_fps", args.fps,
            "--reference", reference, "--offset-output", offsets,
        )

    run(
        "synchronization/export_aligned.py", "--workdir", workdir, "--offsets", offsets,
        "--reference", reference, "--output-dir", output / "aligned", *visualization,
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"{Path(error.cmd[1]).name} failed with exit status {error.returncode}") from None
