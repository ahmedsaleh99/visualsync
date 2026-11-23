"""Extract video frames into the directory layout used by VisualSync."""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare(args):
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    if not source.is_dir():
        raise ValueError(f"Input directory does not exist: {source}")
    if "_" in args.scene or not args.scene:
        raise ValueError("Scene must be nonempty and contain no underscores")
    if args.max_width < 0 or not 1 <= args.jpeg_quality <= 100:
        raise ValueError("max-width must be nonnegative; jpeg-quality must be 1–100")
    if args.sample_fps is not None and (not math.isfinite(args.sample_fps) or args.sample_fps <= 0):
        raise ValueError("sample-fps must be finite and positive")
    videos = sorted(p for p in source.iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"})
    if not videos:
        raise ValueError(f"No videos found in {source}")
    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text()) if source_manifest_path.exists() else {}
    source_records = {record["file"]: record for record in source_manifest.get("clips", {}).values()}
    camera_ids = [
        Path(source_records.get(video.name, {}).get("source_video", {}).get("name", video.name)).stem.split("_")[0]
        for video in videos
    ]
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError("Source clips must have unique camera names before the first underscore")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max-frames must be positive")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output.parent / (output.name + ".manifest.json")
    manifest = {
        "status": "processing", "source_directory": str(source), "workdir": str(output),
        "scene": args.scene, "synchronization_applied": source_manifest.get("use_synchronization", False),
        "temporal_resampling_applied": args.sample_fps is not None,
        "reference_camera": source_manifest.get("main_video", {}).get("name", "").split("_")[0] or None,
        "timestamp_formula": "clip_time_seconds = zero_based_frame_index / fps",
        "views": [],
    }
    thumbnails = []
    cv2.setNumThreads(2)
    for video, camera in zip(videos, camera_ids):
        view_name = f"{args.scene}_cam{camera}"
        view = output / view_name
        rgb = view / "rgb"
        metadata_path = view / "metadata.json"
        stat = video.stat()
        identity = {"source_file": str(video), "source_size": stat.st_size,
                    "source_mtime_ns": stat.st_mtime_ns, "max_width": args.max_width,
                    "jpeg_quality": args.jpeg_quality}
        if args.sample_fps is not None:
            identity["sample_fps"] = args.sample_fps
        if args.max_frames is not None:
            identity["max_frames"] = args.max_frames
        if view.exists():
            if not args.resume or not metadata_path.exists():
                raise ValueError(f"Refusing to overwrite {view}; use --resume for a matching extraction")
            previous = json.loads(metadata_path.read_text())
            if previous.get("extraction") != identity:
                raise ValueError(f"Source or extraction settings changed for {view}; use a new output directory")
            if previous.get("status") == "complete":
                count = previous["frame_count"]
                if len(list(rgb.glob("*.jpg"))) == count and all((rgb / f"{i:06d}.jpg").exists() for i in range(count)):
                    manifest["views"].append(previous)
                    if args.visualize:
                        first = cv2.imread(str(rgb / "000000.jpg"))
                        thumbnails.append((camera, first))
                    print(f"{camera}: reused {count} frames", flush=True)
                    continue
        rgb.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot decode {video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        expected_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            capture.release()
            raise ValueError(f"Invalid FPS for {video}: {fps}")
        output_fps = args.sample_fps if args.sample_fps is not None else fps
        if output_fps > fps:
            capture.release()
            raise ValueError(f"Requested FPS {output_fps} exceeds source FPS {fps} for {video}")
        metadata = {"view": view_name, "camera": camera, "status": "processing",
                    "extraction": identity, "fps": output_fps, "source_fps": fps, "static_camera": True,
                    "frame_name_pattern": "rgb/{zero_based_frame_index:06d}.jpg",
                    "source_clip": source_records.get(video.name)}
        save_json(metadata_path, metadata)
        count = 0
        decoded_count = 0
        next_source_index = 0
        try:
            while capture.grab():
                source_index = decoded_count
                decoded_count += 1
                if source_index != next_source_index or count == args.max_frames:
                    continue
                ok, frame = capture.retrieve()
                if not ok:
                    raise RuntimeError(f"Cannot retrieve frame {source_index} from {video}")
                height, width = frame.shape[:2]
                if count == 0:
                    metadata["source_resolution"] = [width, height]
                if args.max_width and width > args.max_width:
                    frame = cv2.resize(frame, (args.max_width, round(height * args.max_width / width)), interpolation=cv2.INTER_AREA)
                if count == 0:
                    metadata["output_resolution"] = [frame.shape[1], frame.shape[0]]
                    if args.visualize:
                        thumbnails.append((camera, frame.copy()))
                destination = rgb / f"{count:06d}.jpg"
                # Rewrite incomplete runs: an existing JPEG may be truncated.
                if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
                    raise RuntimeError(f"Cannot write {destination}")
                count += 1
                next_source_index = round(count * (fps / output_fps))
                if count % 300 == 0:
                    print(f"{camera}: {count} frames saved", flush=True)
        finally:
            capture.release()
        expected_count = source_records.get(video.name, {}).get("frame_count", expected_count)
        if count == 0 or (expected_count > 0 and decoded_count != expected_count):
            raise RuntimeError(f"Decoded {decoded_count} frames for {camera}, expected {expected_count}; extraction is incomplete")
        metadata.update(status="complete", frame_count=count, duration_seconds=count / output_fps,
                        source_frame_count=decoded_count)
        save_json(metadata_path, metadata)
        manifest["views"].append(metadata)
        save_json(manifest_path, manifest)
        print(f"{camera}: complete, {count} frames at {output_fps:.8f} fps", flush=True)
    if args.deva_compat:
        for metadata in manifest["views"]:
            alias = output / metadata["view"] / "rgb_aligned"
            try:
                alias.symlink_to("rgb", target_is_directory=True)
            except FileExistsError:
                # Preserve existing aliases and separately aligned frame directories.
                if not alias.is_dir():
                    raise
    manifest.update(status="complete", total_frames=sum(v["frame_count"] for v in manifest["views"]))
    save_json(manifest_path, manifest)
    print(f"Prepared {len(videos)} views / {manifest['total_frames']} frames: {output}")
    print(f"Manifest: {manifest_path}")
    if not args.visualize:
        return

    sheet = np.zeros((math.ceil(len(thumbnails) / 4) * 210, 1280, 3), dtype=np.uint8)
    for index, (camera, first) in enumerate(thumbnails):
        if first is None:
            raise RuntimeError(f"Cannot read the first extracted frame for {camera}")
        y, x = divmod(index, 4)
        sheet[y * 210:y * 210 + 180, x * 320:x * 320 + 320] = cv2.resize(first, (320, 180))
        cv2.putText(sheet, camera, (x * 320 + 8, y * 210 + 202), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
    preview = output.parent / (output.name + ".preview.jpg")
    if not cv2.imwrite(str(preview), sheet):
        raise RuntimeError(f"Cannot write preview {preview}")
    print(f"Preview: {preview}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene", required=True, help="Shared scene prefix without underscores; cameras are treated as static")
    parser.add_argument("--max-width", type=int, default=1280, help="Downscale wider frames; 0 preserves original resolution")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--sample-fps", type=float, help="Sample at this rate using the nearest source frame (default: native FPS)")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to save per camera")
    parser.add_argument("--deva-compat", action="store_true",
                        help="Provide DEVA's rgb_aligned input name as a symlink to rgb")
    parser.add_argument("--visualize", action="store_true", help="Save a camera contact sheet")
    parser.add_argument("--resume", action="store_true")
    prepare(parser.parse_args())
