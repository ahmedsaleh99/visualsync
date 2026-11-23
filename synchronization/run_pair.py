"""Run the author's VisualSync pipeline for a pair of prepared camera views."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-step", type=int, default=12)
    parser.add_argument("--interval", type=int, default=10, help="Frame sampling interval")
    parser.add_argument("--offset-range", type=int, default=15, help="Search radius in sampled frames")
    parser.add_argument("--visualize", action="store_true", help="Save tracking and pairwise diagnostic previews")
    parser.add_argument("--fps", type=float, default=1, help="Playback rate of tracking previews")
    parser.add_argument("--reuse-tracks", action="store_true", help="Reuse existing author track archives in the output directory")
    parser.add_argument("views", nargs=2, metavar="VIEW")
    args = parser.parse_args()
    if min(args.grid_step, args.interval, args.offset_range) < 1:
        parser.error("Grid step, interval, and offset range must be positive")

    first, second = args.views
    scene = first.split("_", 1)[0]
    if first == second or second.split("_", 1)[0] != scene:
        parser.error("Select two distinct views from the same scene")
    for name in args.views:
        if Path(name).name != name:
            parser.error(f"Expected a view name, got {name}")
        view = args.workdir / name
        images = sorted((view / "rgb_aligned").glob("*.jpg"))
        masks = sorted((view / "deva" / "Annotations").glob("*.png"))
        if not images or [p.stem for p in images] != [p.stem for p in masks]:
            parser.error(f"Expected matching rgb_aligned frames and DEVA masks in {view}")
        if not (view / "vggt" / "camera_parameters.npz").is_file():
            parser.error(f"Missing VGGT camera parameters in {view}")

    directory = Path(__file__).resolve().parent
    track_root = args.output_dir / "tracks"
    result_root = args.output_dir / "pairs"
    visualization = ["--visualize"] if args.visualize else []

    def run(script, *arguments):
        subprocess.run([sys.executable, str(directory / script), *map(str, arguments)], check=True)

    for name in args.views:
        view = args.workdir / name
        if args.reuse_tracks and (track_root / name / "tracks.pkl").is_file():
            continue
        run(
            "run_cotracker_v5.py", "--video_dir", view / "rgb_aligned",
            "--mask_dir", view / "deva" / "Annotations",
            "--save_dir", track_root / name,
            "--grid_step", args.grid_step, "--interval", args.interval,
            "--disable_blurry", "--disable_seg_filter", "--fps", args.fps, *visualization,
        )

    pair = ["--dataset_root", args.workdir, "--video1_name", first, "--video2_name", second]
    checkpoint = directory.parent / "pretrained" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    checkpoint_args = ["--checkpoint", checkpoint] if checkpoint.is_file() else []
    run(
        "img_match_v4.py", *pair, "--save_root", result_root / scene,
        "--mask_prefix", "deva", "--interval", args.interval, "--batch_size", 1,
        *checkpoint_args,
    )
    try:
        run(
            "filter_corr_v2.py", "--dataset_root", args.workdir,
            "--result_root", result_root, "--track_root", track_root,
            "--result_name1", first, "--result_name2", second, "--mask_prefix", "deva", *visualization,
        )
    except subprocess.CalledProcessError as error:
        if error.returncode != 3:
            raise
        raise SystemExit(3) from None
    pair_output = result_root / scene / f"{first}__{second}"
    if not (pair_output / "tracks_match_v2.npz").is_file():
        print(f"No filtered track correspondences for {first} and {second}")
        raise SystemExit(3)
    run(
        "shaowei_sync_v6.py", *pair, "--result_root", result_root,
        "--use_v2", "--use_vggt", "--disable_gt", "--offset_range", args.offset_range, *visualization,
    )
    print(pair_output / "result_candidates_v3_v2_vggt.pkl")


if __name__ == "__main__":
    main()
