"""Local browser interface for VisualSync when Tk has no display."""

from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit

from long_video_sync import DEFAULT_VIDEO_DIR, export_trimmed, extract_samples, fit_models, VIDEO_EXTENSIONS, sync_samples, trim_plan
from sync_progress import SyncProgress


class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.active_steps = set()
        self.samples = None
        self.analysis = None
        self.directory = None
        self.messages = []
        self.progress = SyncProgress()

    @property
    def busy(self):
        return bool(self.active_steps)

    def can_start(self, step):
        return not self.active_steps or (step == "extract" and self.active_steps == {"sync"})

    def start(self, step, action, *, expected_samples=None):
        with self.lock:
            if not self.can_start(step):
                raise ValueError("An operation is already running.")
            if expected_samples is not None and self.samples is not expected_samples:
                raise ValueError("The selected samples changed. Start synchronization again.")
            if not self.busy:
                self.messages.clear()
            directory = self.directory
            self.active_steps.add(step)
            self.progress.start(step)
        threading.Thread(target=self.run, args=(step, directory, action), daemon=True).start()

    def run(self, step, directory, action):
        def update(completed, total):
            with self.lock:
                self.progress.update(step, completed, total)

        success = False
        try:
            action(self.log, update)
            success = True
        except Exception as error:
            self.log(f"Error: {error}")
        finally:
            with self.lock:
                self.progress.finish(step, success, preserve_sync="sync" in self.active_steps)
                if step == "sync" and directory != self.directory:
                    self.progress.steps_status.pop("sync", None)
                    self.progress.reset_pipeline()
                self.active_steps.remove(step)

    def log(self, message):
        with self.lock:
            self.messages.append(message)

    def update_pipeline(self, *event):
        with self.lock:
            self.progress.update_pipeline(*event)


def serve(port):
    job = Job()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data, content_type="application/json"):
            content = json.dumps(data).encode() if content_type == "application/json" else data
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def local_host(self):
            host = self.headers.get("Host", "").partition(":")[0]
            return host in {"localhost", "127.0.0.1"}

        def do_GET(self):
            if not self.local_host():
                return self.reply(403, {"error": "Use a localhost URL."})
            if self.path == "/":
                return self.reply(200, Path(__file__).with_name("sync_browser.html").read_bytes(),
                                  "text/html; charset=utf-8")
            location = urlsplit(self.path)
            if location.path in {"/videos", "/files"}:
                if location.path == "/videos":
                    root = DEFAULT_VIDEO_DIR.resolve()
                else:
                    with job.lock:
                        root = job.directory
                    if root is None:
                        return self.reply(404, {"error": "Extract samples or open a saved job first."})
                    root = root.resolve()
                raw_folder = parse_qs(location.query).get("folder", ["."])[0]
                directory = (root / raw_folder).resolve()
                if not directory.is_relative_to(root):
                    return self.reply(400, {"error": "Choose a folder inside the displayed tree."})
                if not directory.is_dir():
                    return self.reply(404, {"error": "Video folder does not exist."})
                directories, videos, files = [], [], []
                try:
                    with os.scandir(directory) as entries:
                        for entry in entries:
                            if entry.is_dir(follow_symlinks=False):
                                directories.append((directory / entry.name).relative_to(root).as_posix())
                            elif location.path == "/videos":
                                if Path(entry.name).suffix.lower() in VIDEO_EXTENSIONS and entry.is_file():
                                    videos.append(entry.name)
                            elif entry.is_file(follow_symlinks=False):
                                files.append(entry.name)
                except OSError as error:
                    return self.reply(400, {"error": f"Could not list folder: {error.strerror}"})
                return self.reply(200, {
                    "directory": str(directory), "directories": sorted(directories),
                    "videos": sorted(videos), "files": sorted(files),
                })
            if self.path == "/status":
                with job.lock:
                    result = {"busy": job.busy, "messages": job.messages[:], "analysis": job.analysis,
                              "can_extract": job.can_start("extract"),
                              "progress": job.progress.snapshot(),
                              "pipeline_window": job.progress.pipeline_window,
                              "samples_ready": job.samples is not None,
                              "directory": str(job.directory) if job.directory is not None else None}
                return self.reply(200, result)
            self.reply(404, {"error": "Page not found."})

        def do_POST(self):
            if not self.local_host():
                return self.reply(403, {"error": "Use a localhost URL."})
            try:
                # Only accept the browser's same-origin JSON requests.
                if self.headers.get_content_type() != "application/json":
                    return self.reply(415, {"error": "Expected JSON."})
                origin = self.headers.get("Origin")
                host = self.headers.get("Host")
                if origin and origin not in {f"http://{host}", f"https://{host}"}:
                    return self.reply(403, {"error": "Cross-origin requests are disabled."})
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 1_000_000:
                    raise ValueError("Invalid request size.")
                data = json.loads(self.rfile.read(size))
                result = self.handle_action(data)
            except (ValueError, TypeError, KeyError, OSError) as error:
                return self.reply(400, {"error": str(error)})
            self.reply(200, result or {"ok": True})

        def handle_action(self, data):
            if self.path == "/extract":
                paths = data["paths"]
                if len(paths) < 2:
                    raise ValueError("Enter at least two video paths.")
                if not data["output"].strip():
                    raise ValueError("Enter a results directory.")
                results_folder = Path(data["output"]).expanduser()

                def work(log, progress):
                    result = extract_samples(
                        paths, data.get("reference"), results_folder, int(data["count"]),
                        float(data["duration"]) if data["duration"] else None, float(data["fps"]), log,
                        data.get("camera_fps"), named_run=True, progress=progress,
                    )
                    directory = Path(result["run_directory"])
                    with job.lock:
                        job.samples, job.directory, job.analysis = result, directory, None
                    log(f"Samples ready: {directory / 'samples.json'}")

                job.start("extract", work)
                return
            if self.path == "/sync":
                with job.lock:
                    samples, directory = job.samples, job.directory
                if samples is None:
                    raise ValueError("Extract samples or open samples.json first.")

                def work(log, progress):
                    result = sync_samples(
                        samples, directory, data["objects"], bool(data["visualize"]), log,
                        float(data["radius"]), progress=progress, pipeline_progress=job.update_pipeline,
                        reuse_stages=bool(data.get("reuse_stages", False)),
                    )
                    with job.lock:
                        if job.samples is samples:
                            job.analysis = result
                    succeeded = sum(bool(window["offsets"]) for window in result["windows"])
                    log(f"{succeeded}/{len(result['windows'])} windows yielded offsets. Results: {directory}")

                job.start("sync", work, expected_samples=samples)
                return
            if self.path == "/open":
                path = Path(data["path"]).expanduser().resolve()
                saved = json.loads(path.read_text())
                windows = saved["windows"]
                if not windows:
                    raise ValueError("The manifest has no sample windows.")
                is_result = all("offsets" in window for window in windows)
                if is_result:
                    fit_models(saved, "linear")
                samples_path = path.parent / "samples.json"
                samples = (json.loads(samples_path.read_text()) if samples_path.is_file() else None) if is_result else saved
                with job.lock:
                    if job.busy:
                        raise ValueError("An operation is already running.")
                    job.samples, job.directory = samples, path.parent
                    job.analysis = saved if is_result else None
                    job.messages = [f"Opened {path}"]
                    job.progress.restore(job.samples, job.analysis)
                return

            with job.lock:
                analysis, directory = job.analysis, job.directory
            if analysis is None:
                raise ValueError("Synchronize samples or open saved sync results first.")
            model = fit_models(analysis, data["mode"])
            if self.path == "/model":
                try:
                    plan = trim_plan(model)
                except ValueError as error:
                    plan = {"error": str(error)}
                return {"model": model, "plan": plan}
            elif self.path == "/export":
                trim_plan(model)
                destination = directory / f"trimmed-{model['mode']}-{datetime.now():%Y%m%d-%H%M%S-%f}"

                def work(log, progress):
                    export_trimmed(model, destination, log, progress=progress)
                    log(f"Trim complete. Report: {destination / 'trim_report.json'}")

                job.start("export", work)
            else:
                raise ValueError("Unknown action.")

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"VisualSync browser GUI: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
