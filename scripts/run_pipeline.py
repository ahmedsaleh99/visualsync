"""Run VisualSync on unsynchronized videos sampled at a common frame rate."""

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from pipeline_checkpoints import PipelineCheckpoints


REPOSITORY = Path(__file__).resolve().parents[1]
POSE_FILE = Path("vggt/camera_parameters.npz")
PROGRESS_PREFIX = "VisualSync progress: "


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
    parser.add_argument("--estimate-only", action="store_true", help="Save offsets without rendering aligned output videos")
    parser.add_argument("--visualize", action="store_true", help="Save diagnostic images, overlays, plots, and the camera grid")
    parser.add_argument("--report-progress", action="store_true", help="Emit machine-readable stage progress")
    parser.add_argument("--reuse-stages", action="store_true", help="Reuse compatible completed synchronization stages")
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

    def report(stage, completed, total):
        if args.report_progress:
            print(PROGRESS_PREFIX + json.dumps([stage, completed, total]), flush=True)

    def run_stage(stage, *command, checkpoint=None, **options):
        report(stage, 0, None)
        if checkpoint is None:
            run(*command, **options)
        else:
            settings, files = checkpoint
            if checkpoints.previous(stage, settings):
                print(f"Reusing {stage}: {command[0]}", flush=True)
            else:
                run(*command, **options)
                checkpoints.save(stage, settings, files)
        report(stage, 1, 1)

    run_stage(
        "prepare",
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

    frames = [workdir / name / "rgb" / f"{index:06d}.jpg"
              for name, view in sorted(views.items()) for index in range(view["frame_count"])]
    checkpoints = PipelineCheckpoints(output, frames, args.reuse_stages)
    segmentation = [frame.parent.parent / "gsam2" / "mask" / frame.name for frame in frames]
    if args.visualize:
        segmentation += [frame.parent.parent / "gsam2" / "vis" / frame.name for frame in frames]
    masks = [frame.parent.parent / "deva" / "Annotations" / (frame.stem + ".png") for frame in frames]
    poses = [workdir / view / POSE_FILE for view in views]
    run_stage(
        "segment",
        "preprocess/run_dino_sam2.py", "--workdir", workdir,
        *visualization, "--objects", *args.objects,
        checkpoint=([args.objects, args.visualize], segmentation),
    )
    run_stage(
        "mask_track",
        "evaluation/eval_with_detections.py", "--workdir", workdir,
        "--input-dir", "gsam2", "--output-dir", "deva", "--amp",
        cwd=REPOSITORY / "Tracking-Anything-with-DEVA",
        checkpoint=([], masks),
    )
    colmap = ["--save_colmap"] if args.visualize else []
    run_stage("pose", "preprocess/vggt_to_colmap.py", "--workdir", workdir, "--vis_path", output / "vggt", *colmap,
              checkpoint=([args.visualize], poses))

    grid_step = 48 if max(view["frame_count"] for view in views.values()) > 60 else 16
    track_settings = [grid_step, args.interval, args.fps, args.visualize]
    tracks = [sync / "tracks" / view / "tracks.pkl" for view in views]
    if args.visualize:
        tracks += [sync / "tracks" / view / "tracks.mp4" for view in views]
    saved_tracks = checkpoints.previous("point_track", track_settings)
    if saved_tracks:
        print("Reusing point tracks", flush=True)
    report("point_track", 0, len(views))
    for index, (view, metadata) in enumerate(views.items(), 1):
        directory = workdir / view
        if not (directory / POSE_FILE).is_file():
            raise SystemExit(f"VGGT did not save camera parameters for {view}")
        batch_size = max(1, min(100, 30000 // metadata["frame_count"]))
        if not saved_tracks:
            run(
                "synchronization/run_cotracker_v5.py",
                "--video_dir", directory / "rgb_aligned", "--mask_dir", directory / "deva" / "Annotations",
                "--save_dir", sync / "tracks" / view, "--fps", args.fps,
                "--grid_step", grid_step, "--interval", args.interval,
                "--max_query_per_batch", batch_size, "--feature-chunk-size", 32,
                "--disable_blurry", "--disable_seg_filter", *visualization,
            )
        report("point_track", index, len(views))
    if not saved_tracks:
        checkpoints.save("point_track", track_settings, tracks)

    ordered_views = [reference, *sorted(views.keys() - {reference})]
    with TemporaryDirectory(prefix="visualsync-pairs-") as temporary:
        candidates = Path(temporary)
        scene = manifest["scene"]
        (candidates / scene).mkdir()
        pair_count = math.comb(len(ordered_views), 2)
        pair_settings = [ordered_views, radius, args.visualize]
        saved_pairs = checkpoints.previous("pair_score", pair_settings)
        if saved_pairs:
            print("Reusing camera pair scores", flush=True)
        accepted_pairs = set(saved_pairs["pairs"]) if saved_pairs else set()
        report("pair_score", 0, pair_count)
        for index, (first, second) in enumerate(combinations(ordered_views, 2), 1):
            pair = f"{first}__{second}"
            if not saved_pairs:
                print(f"Scoring {first} against {second}", flush=True)
                completed = run(
                    "synchronization/run_pair.py", "--workdir", workdir, "--output-dir", sync,
                    "--grid-step", grid_step, "--interval", args.interval, "--offset-range", radius,
                    "--fps", args.fps, "--reuse-tracks", *visualization, first, second, check=False,
                )
                if completed.returncode != 3:
                    completed.check_returncode()
                    accepted_pairs.add(pair)
            if pair in accepted_pairs:
                (candidates / scene / pair).symlink_to(sync / "pairs" / scene / pair, target_is_directory=True)
            else:
                (candidates / scene / pair).mkdir()
            report("pair_score", index, pair_count)
        if not saved_pairs:
            pairs = sorted(accepted_pairs)
            files = [sync / "pairs" / scene / pair / "result_candidates_v3_v2_vggt.pkl" for pair in pairs]
            checkpoints.save("pair_score", pair_settings, files, pairs=pairs)

        offsets = output / "author_offsets.json"
        run_stage(
            "solve",
            "synchronization/eval_result3.py", "--result_roots", candidates, "--use_vggt",
            "--disable_gt", "--est_choice", "irls", "--dataset_fps", args.fps,
            "--reference", reference, "--offset-output", offsets,
            checkpoint=([reference, args.fps], [offsets]),
        )

    if args.estimate_only:
        return

    run(
        "synchronization/export_aligned.py", "--workdir", workdir, "--offsets", offsets,
        "--reference", reference, "--output-dir", output / "aligned", *visualization,
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"{Path(error.cmd[1]).name} failed with exit status {error.returncode}") from None
