"""Desktop interface for long-recording VisualSync jobs."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    tk = None

from long_video_sync import DEFAULT_VIDEO_DIR, default_reference, export_trimmed, extract_samples, fit_models, list_videos, source_time, sync_samples, trim_plan
from sync_progress import PIPELINE_STAGES, SyncProgress


class SyncWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("VisualSync · Long recordings")
        self.root.geometry("960x780")
        self.root.minsize(760, 640)
        self.paths = []
        self.samples = None
        self.analysis = None
        self.run_dir = None
        self.busy = False
        self.events = queue.Queue()
        self.progress = SyncProgress()
        self.progress_widgets = {}
        self.rendered_progress = None
        self.active_step = None
        self.reference = tk.StringVar()
        self.chosen_reference = None
        self.output = tk.StringVar(value=str(Path.cwd() / "data" / "long-sync"))
        self.count = tk.StringVar(value="5")
        self.duration = tk.StringVar(value="60")
        self.fps = tk.StringVar(value="5")
        self.radius = tk.StringVar(value="20")
        self.objects = tk.StringVar(value="person")
        self.visualize = tk.BooleanVar(value=False)
        self.reuse_stages = tk.BooleanVar(value=True)
        self.mode = tk.StringVar(value="linear")
        self.camera = tk.StringVar()
        self.status = tk.StringVar(value="Choose the original camera videos.")
        self.summary = tk.StringVar()
        self.actions = []

        body = ttk.Frame(root, padding=16)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(8, weight=1)
        body.rowconfigure(12, weight=1)
        ttk.Label(body, text="Long-recording synchronization", font=("TkDefaultFont", 16, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 12)
        )
        self.button(body, "Choose videos…", self.choose_videos, 1, 0)
        ttk.Label(body, textvariable=self.status, wraplength=500).grid(row=1, column=1, sticky="w")
        self.button(body, "Use a video folder…", self.choose_video_folder, 1, 2)
        ttk.Label(body, text="Reference video").grid(row=2, column=0, sticky="w")
        self.references = ttk.Combobox(body, textvariable=self.reference, state="readonly")
        self.references.grid(row=2, column=1, columnspan=2, sticky="ew", pady=3)
        self.references.bind("<<ComboboxSelected>>", self.choose_reference)
        ttk.Label(body, text="Results folder").grid(row=3, column=0, sticky="w")
        ttk.Entry(body, textvariable=self.output).grid(row=3, column=1, sticky="ew", padx=(0, 8))
        self.button(body, "Browse…", self.choose_output, 3, 2)

        settings = ttk.Frame(body)
        settings.grid(row=4, column=0, columnspan=3, sticky="w", pady=10)
        for column, (label, variable, width) in enumerate((
            ("Windows across reference", self.count, 6), ("Window duration (sec)", self.duration, 8),
            ("Sample FPS", self.fps, 9), ("Search radius (s)", self.radius, 8),
            ("Object prompts (comma-separated)", self.objects, 25),
        )):
            ttk.Label(settings, text=label).grid(row=0, column=column, sticky="w", padx=(0, 14))
            if variable is self.duration:
                field = ttk.Spinbox(settings, from_=1, to=86400, increment=1, textvariable=variable, width=width)
            else:
                field = ttk.Entry(settings, textvariable=variable, width=width)
            field.grid(row=1, column=column, sticky="w")
        ttk.Checkbutton(body, text="Save snapshot diagnostic previews", variable=self.visualize).grid(
            row=5, column=0, sticky="w"
        )
        ttk.Checkbutton(body, text="Reuse completed stages from this run", variable=self.reuse_stages).grid(
            row=5, column=1, columnspan=2, sticky="w"
        )
        controls = ttk.Frame(body)
        controls.grid(row=6, column=0, columnspan=3, sticky="ew", pady=8)
        self.add_progress(controls, "extract", 0, row=0)
        self.button(controls, "1. Extract samples", self.run_extraction, 2, 0)
        self.sync = self.button(controls, "2. Synchronize samples", self.run_analysis, 2, 1)
        self.button(controls, "Open samples or results…", self.load_analysis, 2, 2)
        self.add_progress(controls, "sync", 1, row=3)
        self.pipeline_panel = ttk.LabelFrame(controls, text="Synchronization pipeline")
        self.pipeline_panel.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        self.pipeline_panel.grid_remove()
        for index, (stage, name) in enumerate(PIPELINE_STAGES.items()):
            row, column = divmod(index, 4)
            ttk.Label(self.pipeline_panel, text=name).grid(row=row * 3, column=column, sticky="w", padx=4)
            self.add_progress(self.pipeline_panel, stage, column, row=row * 3 + 1)

        model_controls = ttk.Frame(body)
        model_controls.grid(row=7, column=0, columnspan=3, sticky="ew")
        ttk.Label(model_controls, text="Drift model").pack(side="left", padx=(0, 6))
        model = ttk.Combobox(model_controls, textvariable=self.mode, values=("linear", "piecewise"),
                             state="readonly", width=12)
        model.pack(side="left", padx=(0, 14))
        model.bind("<<ComboboxSelected>>", self.refresh)
        ttk.Label(model_controls, text="Camera").pack(side="left", padx=(0, 6))
        self.cameras = ttk.Combobox(model_controls, textvariable=self.camera, state="readonly", width=38)
        self.cameras.pack(side="left", fill="x", expand=True)
        self.cameras.bind("<<ComboboxSelected>>", self.refresh)
        self.chart = tk.Canvas(body, height=175, background="#f5f7fa", highlightthickness=0)
        self.chart.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=8)
        self.chart.bind("<Configure>", self.refresh)
        ttk.Label(body, textvariable=self.summary, wraplength=850).grid(
            row=9, column=0, columnspan=3, sticky="w"
        )

        exports = ttk.Frame(body)
        exports.grid(row=10, column=0, columnspan=3, sticky="w", pady=8)
        self.save = self.button(exports, "Save sync map…", self.save_map, 0, 0)
        self.export = self.button(exports, "3. Export trim only…", self.run_export, 0, 1)
        self.add_progress(exports, "export", 1)
        ttk.Label(body, text="Run log").grid(row=11, column=0, sticky="w")
        log_frame = ttk.Frame(body)
        log_frame.grid(row=12, column=0, columnspan=3, sticky="nsew")
        self.log = tk.Text(log_frame, height=7, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.set_busy(False)
        self.root.after(100, self.read_events)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def button(self, parent, text, command, row, column):
        button = ttk.Button(parent, text=text, command=command)
        button.grid(row=row, column=column, sticky="w", padx=(0, 8), pady=3)
        self.actions.append(button)
        return button

    def choose_videos(self):
        paths = filedialog.askopenfilenames(
            initialdir=DEFAULT_VIDEO_DIR,
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi *.webm *.mts *.m4v"), ("All files", "*")]
        )
        if paths:
            self.set_videos(paths)

    def choose_video_folder(self):
        path = filedialog.askdirectory(initialdir=DEFAULT_VIDEO_DIR)
        if not path:
            return
        videos = list_videos(Path(path))
        if not videos:
            messagebox.showinfo("Video folder", "Choose a folder that directly contains videos.")
            return
        self.set_videos([str(video) for video in videos])

    def set_videos(self, paths):
        self.paths = list(paths)
        self.references.configure(values=paths)
        if self.chosen_reference not in paths:
            self.chosen_reference = None
        self.reference.set(self.chosen_reference or default_reference(paths))
        self.status.set(f"{len(paths)} videos selected")

    def choose_reference(self, _event):
        self.chosen_reference = self.reference.get()

    def choose_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output.set(path)

    def set_busy(self, busy):
        self.busy = busy
        for button in self.actions:
            button.configure(state="disabled" if busy else "normal")
        if self.samples is None:
            self.sync.configure(state="disabled")
        for button in (self.save, self.export):
            if self.analysis is None:
                button.configure(state="disabled")

    def add_progress(self, parent, step, column, *, row=1):
        bar = ttk.Progressbar(parent, length=140)
        bar.grid(row=row, column=column, sticky="ew", padx=4)
        label = ttk.Label(parent)
        label.grid(row=row + 1, column=column, sticky="w", padx=4)
        self.progress_widgets[step] = (bar, label)

    def show_progress(self):
        steps = self.progress.snapshot()
        window = self.progress.pipeline_window
        if (steps, window) == self.rendered_progress:
            return
        self.rendered_progress = (steps, window)
        if window:
            index, total = window
            self.pipeline_panel.configure(text=f"Synchronization pipeline · Window {index}/{total}")
            self.pipeline_panel.grid()
        else:
            self.pipeline_panel.grid_remove()
        for step, (bar, label) in self.progress_widgets.items():
            update = steps.get(step, {})
            state = update.get("state")
            completed, total = update.get("completed", 0), update.get("total")
            unit = {"extract": "samples", "sync": "windows", "export": "videos",
                    "point_track": "cameras", "pair_score": "pairs"}.get(step)
            parts = []
            if total:
                parts.append(f"{completed}/{total}" + (f" {unit}" if unit else ""))
            elapsed = update.get("elapsed_seconds")
            if elapsed is not None:
                minutes, seconds = divmod(elapsed, 60)
                parts.append(f"{minutes:02d}:{seconds:02d}")
            if state == "running" and total is None:
                if bar.cget("mode") != "indeterminate":
                    bar.configure(mode="indeterminate")
                    bar.start()
            else:
                bar.stop()
                bar.configure(mode="determinate", maximum=total or 1, value=completed)
            label.configure(text=" · ".join(parts))

    def write_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def worker(self, step, action, result_event):
        self.active_step = step
        self.progress.start(step)
        self.show_progress()
        self.set_busy(True)

        def run():
            def update(completed, total):
                self.events.put(("progress", (completed, total)))

            try:
                result = action(lambda text: self.events.put(("log", text)), update)
                self.events.put((result_event, result))
            except Exception as error:
                self.events.put(("error", str(error)))

        threading.Thread(target=run, daemon=True).start()

    def run_extraction(self):
        try:
            paths = list(self.paths)
            reference = self.reference.get()
            count, fps = int(self.count.get()), float(self.fps.get())
            duration_text = self.duration.get().strip()
            duration = float(duration_text) if duration_text else None
            if not self.output.get().strip() or len(paths) < 2:
                raise ValueError("Choose at least two videos and a results folder.")
        except ValueError as error:
            messagebox.showerror("Snapshot settings", str(error))
            return
        results_folder = Path(self.output.get()).expanduser()

        def extract(log, progress):
            samples = extract_samples(paths, reference, results_folder, count, duration, fps, log,
                                      named_run=True, progress=progress)
            return samples, Path(samples["run_directory"])

        self.worker("extract", extract, "samples")

    def accept_samples(self, samples, directory):
        self.samples, self.run_dir, self.analysis = samples, directory, None
        self.cameras.configure(values=())
        self.camera.set("")
        self.summary.set("Samples ready for synchronization.")
        self.refresh()
        self.write_log(f"Samples ready: {directory / 'samples.json'}")
        self.set_busy(False)

    def run_analysis(self):
        try:
            radius = float(self.radius.get())
            objects = [item.strip() for item in self.objects.get().split(",") if item.strip()]
        except ValueError as error:
            messagebox.showerror("Synchronization settings", str(error))
            return
        samples, directory, visualize = self.samples, self.run_dir, self.visualize.get()
        reuse_stages = self.reuse_stages.get()

        def pipeline_progress(*event):
            self.events.put(("pipeline_progress", event))

        def synchronize(log, progress):
            return sync_samples(samples, directory, objects, visualize, log, radius,
                                progress=progress, pipeline_progress=pipeline_progress,
                                reuse_stages=reuse_stages)

        self.worker("sync", synchronize, "analysis")

    def load_analysis(self):
        path = filedialog.askopenfilename(filetypes=[("Snapshot JSON", "*.json")])
        if not path:
            return
        try:
            saved = json.loads(Path(path).read_text())
            directory = Path(path).parent
            windows = saved["windows"]
            if not windows:
                raise ValueError("The manifest has no sample windows.")
            if all("offsets" in window for window in windows):
                fit_models(saved, self.mode.get())
                samples_path = directory / "samples.json"
                self.samples = json.loads(samples_path.read_text()) if samples_path.is_file() else None
                self.run_dir = directory
                self.accept_analysis(saved)
            else:
                self.accept_samples(saved, directory)
            self.progress.restore(self.samples, self.analysis)
            self.show_progress()
        except (OSError, ValueError, KeyError, TypeError) as error:
            messagebox.showerror("Open snapshots", str(error))

    def accept_analysis(self, analysis):
        self.analysis = analysis
        cameras = [f"{key} · {Path(data['path']).name}" for key, data in analysis["cameras"].items()
                   if key != analysis["reference"]]
        self.cameras.configure(values=cameras)
        self.camera.set(cameras[0] if cameras else "")
        succeeded = sum(bool(window["offsets"]) for window in analysis["windows"])
        self.write_log(f"{succeeded}/{len(analysis['windows'])} windows yielded offsets. Results: {self.run_dir}")
        self.set_busy(False)
        self.refresh()

    def read_events(self):
        while True:
            try:
                event, value = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "log":
                self.write_log(value)
                continue
            if event == "progress":
                self.progress.update(self.active_step, *value)
                continue
            if event == "pipeline_progress":
                self.progress.update_pipeline(*value)
                continue
            self.progress.finish(self.active_step, event != "error")
            self.active_step = None
            self.show_progress()
            if event == "samples":
                self.accept_samples(*value)
            elif event == "analysis":
                self.accept_analysis(value)
            else:
                self.set_busy(False)
                if event == "error":
                    self.write_log(value)
                    messagebox.showerror("VisualSync", value)
                else:
                    self.write_log(f"Trim complete. See {value}.")
        if self.active_step:
            self.show_progress()
        self.root.after(100, self.read_events)

    def refresh(self, _event=None):
        self.chart.delete("all")
        if not self.analysis or not self.camera.get():
            return
        name = self.camera.get().split(" · ", 1)[0]
        sync_map = fit_models(self.analysis, self.mode.get())
        model = sync_map["models"].get(name)
        samples = [(window["time"], window["offsets"][name]) for window in self.analysis["windows"]
                   if name in window["offsets"]]
        if model is None:
            self.summary.set(f"{len(samples)} usable measurements. More reliable windows or a monotone fit are required.")
        else:
            residual = max(map(abs, model["residuals"]), default=0)
            self.summary.set(f"{len(samples)} usable measurements · largest fit residual {residual:.3f} s")
        if not samples:
            return
        width, height = self.chart.winfo_width(), self.chart.winfo_height()
        if width < 140 or height < 70:
            return
        end = self.analysis["cameras"][sync_map["reference"]]["duration"]
        extended = [] if model is None else [
            (time, source_time(model, time) - time) for time in (0, *model["times"], end)
        ]
        offsets = [value for _, value in samples + extended]
        minimum, maximum = min(offsets), max(offsets)
        span = max(maximum - minimum, 0.01)

        def position(time, offset):
            return (64 + time / end * (width - 86),
                    height - 31 - (offset - minimum + 0.08 * span) / (1.16 * span) * (height - 57))

        self.chart.create_line(64, 12, 64, height - 31, width - 22, height - 31, fill="#64748b")
        self.chart.create_text(64, height - 12, text="0 s", anchor="w")
        self.chart.create_text(width - 22, height - 12, text=f"{end:.1f} s", anchor="e")
        for value in (minimum, maximum) if maximum > minimum else (minimum,):
            self.chart.create_text(59, position(0, value)[1], anchor="e", text=f"{value:.3f} s")
        if extended:
            self.chart.create_line(*(coordinate for point in extended for coordinate in position(*point)),
                                   fill="#2563eb", width=2, dash=(5, 3))
            self.chart.create_line(*(coordinate for time, offset in zip(model["times"], model["offsets"])
                                     for coordinate in position(time, offset)), fill="#2563eb", width=2)
        for time, value in samples:
            x, y = position(time, value)
            self.chart.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#b45309", outline="white")

    def save_map(self):
        path = filedialog.asksaveasfilename(initialdir=self.run_dir, initialfile="sync_map.json",
                                            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            try:
                Path(path).write_text(json.dumps(fit_models(self.analysis, self.mode.get()), indent=2) + "\n")
                self.write_log(f"Saved {path}")
            except OSError as error:
                messagebox.showerror("Save sync map", str(error))

    def run_export(self):
        sync_map = fit_models(self.analysis, self.mode.get())
        try:
            plan = trim_plan(sync_map)
        except ValueError as error:
            messagebox.showerror("Trim export", str(error))
            return
        residual = max(view["residual_drift"] for view in plan["views"].values())
        extrapolation = any(view["extrapolates"] for view in plan["views"].values())
        details = (f"Shared reference span: {plan['start']:.2f}–{plan['end']:.2f} s.\n"
                   f"Maximum modeled drift after trimming: {residual:.3f} s, plus keyframe rounding.\n"
                   "Export copies the original video and audio streams into MKV files.")
        if extrapolation:
            details += "\nThe model extrapolates beyond one or more camera's reliable windows."
        if messagebox.askokcancel("Confirm stream-copy export", details):
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            destination = self.run_dir / f"trimmed-{sync_map['mode']}-{stamp}"

            def export(log, progress):
                export_trimmed(sync_map, destination, log, progress=progress)
                return destination / "trim_report.json"

            self.worker("export", export, "export")

    def close(self):
        if self.busy:
            messagebox.showinfo("VisualSync is running", "Keep the window open until the current operation finishes.")
        else:
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", action="store_true", help="Open the browser interface")
    parser.add_argument("--port", type=int, default=7861, help="Local browser interface port")
    args = parser.parse_args()
    from sync_browser import serve

    if args.browser or tk is None:
        serve(args.port)
        return
    try:
        root = tk.Tk()
    except tk.TclError:
        print("Desktop display unavailable. Starting the browser interface.", flush=True)
        serve(args.port)
        return
    SyncWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
