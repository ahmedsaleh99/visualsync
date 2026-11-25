"""Sample long recordings, fit VisualSync offsets, and stream-copy original clips."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from run_pipeline import PROGRESS_PREFIX

DEFAULT_VIDEO_DIR = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mts", ".m4v"}
PIPELINE = Path(__file__).with_name("run_pipeline.py").resolve()


def probe(path, *options):
    result = subprocess.run(
        ["ffprobe", "-v", "error", *options, "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def inspect_video(path):
    data = probe(path, "-select_streams", "v:0", "-show_streams", "-show_format")
    if not data.get("streams"):
        raise ValueError(f"No video stream in {path}")
    stream = data["streams"][0]
    container = data["format"]
    origin = float(container.get("start_time", 0))
    start = max(0.0, float(stream.get("start_time", origin)) - origin)
    stream_duration = stream.get("duration")
    if stream_duration not in (None, "N/A"):
        duration = start + float(stream_duration)
    else:
        duration = float(container.get("duration", 0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"No usable video duration in {path}")
    return {
        "path": str(Path(path).resolve()),
        "duration": duration,
        "fps": stream.get("avg_frame_rate"),
        "timestamp_origin": origin,
        "video_start": start,
    }


def sample_windows(duration, count, length, fps):
    if not all(math.isfinite(value) and value > 0 for value in (duration, length, fps)) or count < 2:
        raise ValueError("Use at least two windows and positive finite duration and FPS.")
    if length >= duration:
        raise ValueError(f"Window duration must be under {duration:.3f} s for the reference video. Clear the field to choose automatically.")
    frames = math.floor(length * fps + 1e-9)
    if frames < 2:
        raise ValueError("Each snapshot needs at least two frames at the analysis FPS.")
    return np.linspace(0, duration - length, count).tolist(), frames


def list_videos(directory):
    return [path for path in sorted(directory.iterdir())
            if path.suffix.lower() in VIDEO_EXTENSIONS and path.is_file()]


def camera_inputs(paths):
    paths = [Path(path).resolve() for path in paths]
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError("Select at least two distinct input videos.")
    return {f"view{index:02d}": inspect_video(path) for index, path in enumerate(paths, 1)}


def save_manifest(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def default_reference(paths):
    return next((path for path in paths if Path(path).name.casefold().startswith("tripod")), paths[0])


def extract_samples(paths, reference_path, output, count, length, fps, log, camera_fps=None, *,
                    named_run=False, progress=None):
    cameras = camera_inputs(paths)
    reference_path = reference_path or default_reference(paths)
    reference = next((key for key, camera in cameras.items()
                      if camera["path"] == str(Path(reference_path).resolve())), None)
    if reference is None:
        raise ValueError("Select one of the input videos as the reference.")
    for name, camera in cameras.items():
        stem = Path(camera["path"]).stem.encode("utf-8")[:200].decode("utf-8", errors="ignore")
        camera["sample_filename"] = f"{name}_{stem}.mp4"
    first = cameras[reference]["video_start"]
    last = cameras[reference]["duration"]
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Default sample FPS must be positive and finite.")
    overrides = {str(Path(path).resolve()): value for path, value in (camera_fps or {}).items()}
    rates = {}
    for name, camera in cameras.items():
        value = float(overrides.get(camera["path"], fps))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid sample FPS for {Path(camera['path']).name}.")
        rate = Fraction(str(value)).limit_denominator(100_000)
        if rate <= 0:
            raise ValueError(f"Sample FPS is too low for {Path(camera['path']).name}.")
        rates[name] = rate
        camera["sample_fps"] = float(rate)
    fps = float(min(rates.values()))
    log(f"Synchronization FPS: {fps:g} (lowest sample FPS)")
    span = last - first
    if length is None:
        length = min(60, span / 2)
        log(f"Automatic window duration: {length:.2f} s")
    starts, frames = sample_windows(span, count, length, fps)
    total = len(cameras) * len(starts)
    if progress:
        progress(0, total)
    output = Path(output).resolve()
    if named_run:
        minimum, maximum = float(min(rates.values())), float(max(rates.values()))
        rate_label = f"{minimum:g}" if minimum == maximum else f"{minimum:g}-{maximum:g}"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output /= f"run-{count}samples-{length:g}s-{rate_label}fps-{stamp}"
    output.mkdir(parents=True, exist_ok=not named_run)
    records = []
    samples = {
        "reference": reference, "analysis_fps": fps, "window_duration_seconds": length,
        "cameras": cameras, "windows": records, "run_directory": str(output),
    }
    for index, offset in enumerate(starts, 1):
        start = first + offset
        label = f"window-{index:03d}"
        inputs = output / label / "clips"
        inputs.mkdir(parents=True)
        record = {
            "label": label, "start": start, "time": start + (frames - 1) / (2 * fps),
            "participating_cameras": [],
        }
        records.append(record)

    def extract_clip(record, name):
        label = record["label"]
        start = record["start"]
        camera = cameras[name]
        rate = rates[name]
        clip_frames = math.floor(length * float(rate) + 1e-9)
        if start < camera["video_start"] - 1e-6 or start + clip_frames / float(rate) > camera["duration"] + 1e-6:
            return None
        clip = output / label / "clips" / camera["sample_filename"]
        try:
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start),
                 "-i", camera["path"], "-map", "0:v:0", "-an",
                 "-vf", f"fps={rate}:start_time=0", "-frames:v", str(clip_frames),
                 "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(clip)],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as error:
            raise ValueError(f"Could not extract {name} in {label}: {error.stderr.strip()}") from error
        stream = probe(clip, "-select_streams", "v:0", "-show_entries", "stream=nb_frames")["streams"][0]
        if int(stream.get("nb_frames", 0)) != clip_frames:
            raise ValueError(f"{name} in {label} did not contain the requested {clip_frames} frames.")
        return clip

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(extract_clip, record, name): (record, name)
            for record in records for name in cameras
        }
        try:
            for completed, future in enumerate(as_completed(futures), 1):
                clip = future.result()
                record, name = futures[future]
                if clip is None:
                    log(f"{record['label']}: {Path(cameras[name]['path']).name} does not cover this window.")
                else:
                    record["participating_cameras"].append(name)
                    log(f"Extracted {clip.name} in {record['label']} ({completed}/{total})")
                if progress:
                    progress(completed, total)
        finally:
            for future in futures:
                future.cancel()
    for record in records:
        participants = set(record["participating_cameras"])
        record["participating_cameras"] = [name for name in cameras if name in participants]
    save_manifest(output / "samples.json", samples)
    return samples


def sync_samples(samples, output, objects, visualize, log, max_offset_seconds=20, *,
                 progress=None, pipeline_progress=None, reuse_stages=False):
    if not math.isfinite(max_offset_seconds) or max_offset_seconds <= 0:
        raise ValueError("The pairwise search radius must be positive and finite.")
    if not objects:
        raise ValueError("Enter at least one object prompt.")
    reference, fps, cameras = (samples[key] for key in ("reference", "analysis_fps", "cameras"))
    output = Path(output).resolve()
    records = []
    result = {**samples, "windows": records}
    windows = samples["windows"]
    if progress:
        progress(0, len(windows))
    for index, sample in enumerate(windows, 1):
        if pipeline_progress:
            pipeline_progress(index, len(windows))
        log(f"Synchronizing window {index}/{len(windows)} at {sample['start']:.2f} s")
        window = output / sample["label"]
        inputs = window / "clips"
        record = {**sample, "offsets": {}}
        try:
            participants = sample.get("participating_cameras", cameras)
            if reference not in participants or len(participants) < 2:
                raise ValueError("This window has no overlapping camera pair with the reference.")
            missing = [
                name for name in participants
                if not (inputs / cameras[name].get("sample_filename", f"{name}.mp4")).is_file()
            ]
            if missing:
                raise ValueError("Missing sample clips: " + ", ".join(missing))
            command = [
                sys.executable, "-u", str(PIPELINE), "--input-dir", str(inputs),
                "--output-dir", str(window / "result"), "--fps", str(fps),
                "--reference", reference, "--estimate-only",
                "--max-offset-seconds", str(max_offset_seconds), "--objects", *objects,
            ]
            if visualize:
                command.append("--visualize")
            if reuse_stages:
                command.append("--reuse-stages")
            if pipeline_progress:
                command.append("--report-progress")
            with (window / "pipeline.log").open("w") as trace:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True) as process:
                    for line in process.stdout:
                        trace.write(line)
                        if pipeline_progress and line.startswith(PROGRESS_PREFIX):
                            pipeline_progress(index, len(windows), *json.loads(line[len(PROGRESS_PREFIX):]))
                        elif line.startswith(("Running ", "Scoring ", "Reusing ")):
                            log(line.rstrip())
                    if process.wait():
                        raise ValueError(f"VisualSync failed. See {window / 'pipeline.log'}")
            offsets = json.loads((window / "result" / "author_offsets.json").read_text())
            component = next((group["offset_frames"] for group in offsets["scenes"].get("capture", [])
                              if group["reference"] == f"capture_cam{reference}"), {})
            anchor = component.get(f"capture_cam{reference}")
            if anchor is not None:
                record["offsets"] = {name: (anchor - component[f"capture_cam{name}"]) / fps
                                     for name in participants if component.get(f"capture_cam{name}") is not None}
            if not record["offsets"]:
                raise ValueError("No camera pair could be connected to the reference.")
        except (ValueError, OSError) as error:
            record["error"] = str(error)
            log(f"Window {index}: {error}")
        records.append(record)
        save_manifest(output / "snapshots.json", result)
        if progress:
            progress(index, len(windows))
    return result


def fit_models(analysis, mode):
    if mode not in {"linear", "piecewise"}:
        raise ValueError(f"Unknown drift model: {mode}")
    reference = analysis["reference"]
    models = {}
    for name in analysis["cameras"]:
        samples = [(window["time"], window["offsets"][name])
                   for window in analysis["windows"] if name in window["offsets"]]
        if name == reference:
            end = analysis["cameras"][name]["duration"]
            models[name] = {"times": [0, end], "offsets": [0, 0], "samples": samples}
            continue
        minimum = 2 if mode == "linear" else 3
        if len(samples) < minimum:
            continue
        times, offsets = np.asarray(samples).T
        if mode == "linear":
            design = np.column_stack((np.ones(len(times)), times - times[0]))
            weights = np.ones(len(times))
            threshold = 1 / analysis["analysis_fps"]
            for _ in range(20):
                weighted = np.sqrt(weights)
                coefficients = np.linalg.lstsq(design * weighted[:, None], offsets * weighted, rcond=None)[0]
                residuals = np.abs(offsets - design @ coefficients)
                weights = threshold / np.maximum(threshold, residuals)
            fitted = design @ coefficients
            knots = [0, -1]
            model_times, model_offsets = times[knots], fitted[knots]
        else:
            model_times, model_offsets = times, offsets
        if np.any(np.diff(model_times + model_offsets) <= 0):
            continue
        models[name] = {
            "times": model_times.tolist(), "offsets": model_offsets.tolist(),
            "samples": samples, "residuals": (offsets - np.interp(times, model_times, model_offsets)).tolist(),
        }
    return {"mode": mode, "mapping": "source_seconds = reference_seconds + offset_seconds",
            "reference": reference, "analysis_fps": analysis["analysis_fps"],
            "cameras": analysis["cameras"], "models": models}


def interpolate(value, x, y):
    index = max(0, min(len(x) - 2, int(np.searchsorted(x, value, side="right")) - 1))
    fraction = (value - x[index]) / (x[index + 1] - x[index])
    return y[index] + fraction * (y[index + 1] - y[index])


def source_time(model, time):
    return time + interpolate(time, model["times"], model["offsets"])


def trim_plan(sync_map):
    cameras = sync_map["cameras"]
    models = sync_map["models"]
    missing = cameras.keys() - models.keys()
    if missing:
        raise ValueError("No valid model for: " + ", ".join(Path(cameras[name]["path"]).name for name in sorted(missing)))
    start = 0.0
    end = cameras[sync_map["reference"]]["duration"]
    for name, model in models.items():
        times = model["times"]
        mapped = [time + offset for time, offset in zip(times, model["offsets"])]
        start = max(start, interpolate(cameras[name].get("video_start", 0), mapped, times))
        end = min(end, interpolate(cameras[name]["duration"], mapped, times))
    if end <= start:
        raise ValueError("The fitted cameras have no shared time span.")
    details = {}
    for name, model in models.items():
        cut = max(0.0, source_time(model, start))
        end = min(end, start + cameras[name]["duration"] - cut)
        details[name] = {"requested_cut": cut}
    if end <= start:
        raise ValueError("No common span can be exported with trim only.")
    for name, model in models.items():
        points = [start, *[t for t in model["times"] if start < t < end], end]
        initial = details[name]["requested_cut"] - start
        details[name].update(
            residual_drift=max(abs(source_time(model, t) - t - initial) for t in points),
            extrapolates=start < model["times"][0] or end > model["times"][-1],
        )
    return {"start": start, "end": end, "views": details}


def preceding_keyframe(camera, requested):
    origin = camera["timestamp_origin"]
    seek = requested + origin
    packets = probe(
        camera["path"], "-select_streams", "v:0", "-read_intervals", f"{seek:.9f}%+#32",
        "-show_packets", "-show_entries", "packet=pts_time,flags",
    ).get("packets", [])
    nearby = [float(packet["pts_time"]) - origin for packet in packets
              if "K" in packet.get("flags", "") and packet.get("pts_time") not in (None, "N/A")]
    eligible = [time for time in nearby if time <= requested + 1e-6]
    if eligible:
        return max(0.0, max(eligible))

    # Fall back to decoding keyframe timestamps if the container has no usable seek index.
    data = probe(camera["path"], "-select_streams", "v:0", "-skip_frame", "nokey",
                 "-show_frames", "-show_entries", "frame=best_effort_timestamp_time")
    times = [float(frame["best_effort_timestamp_time"]) - camera["timestamp_origin"]
             for frame in data["frames"] if "best_effort_timestamp_time" in frame]
    if not times:
        raise ValueError(f"No seekable keyframes in {camera['path']}")
    eligible = [time for time in times if time <= requested + 1e-6]
    return max(0.0, max(eligible, default=min(times)))


def export_trimmed(sync_map, output, log, *, progress=None):
    """Trim around the fitted start using stream copy and record keyframe errors."""
    plan = trim_plan(sync_map)
    plan.update(reference=sync_map["reference"], model=sync_map["mode"])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cameras = sync_map["cameras"]
    views = plan["views"]
    if progress:
        progress(0, len(views))
    duration = plan["end"] - plan["start"]
    for name, detail in views.items():
        camera = cameras[name]
        cut = preceding_keyframe(camera, detail["requested_cut"])
        detail["selected_keyframe"] = cut
        detail["keyframe_delta"] = cut - detail["requested_cut"]
        duration = min(duration, camera["duration"] - cut)
    if duration <= 0:
        raise ValueError("Keyframe cuts leave no common export duration.")
    plan["export_duration"] = duration
    for index, (name, detail) in enumerate(views.items(), 1):
        camera = cameras[name]
        path = output / f"{name}-{Path(camera['path']).stem}.mkv"
        log(f"Copying {path.name} at source time {detail['selected_keyframe']:.3f} s")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-n",
             "-ss", f"{detail['selected_keyframe']:.9f}", "-i", camera["path"],
             "-t", f"{duration:.9f}", "-map", "0:v:0", "-map", "0:a?",
             "-c", "copy", "-avoid_negative_ts", "make_zero", str(path)],
            check=True,
        )
        exported = inspect_video(path)
        detail.update(
            source=camera["path"], output=str(path), output_fps=exported["fps"],
            output_duration=exported["duration"],
            output_timestamp_origin=exported["timestamp_origin"],
            output_video_start=exported["video_start"],
        )
        log(f"{path.name}: cut delta {detail['keyframe_delta']:+.3f} s; "
            f"modeled drift up to {detail['residual_drift']:.3f} s")
        if progress:
            progress(index, len(views))
    (output / "trim_report.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan
