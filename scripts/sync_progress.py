"""Progress for the three long-recording workflow steps."""

from time import monotonic


PIPELINE_STAGES = {
    "prepare": "Prepare frames",
    "segment": "Object segmentation",
    "mask_track": "Mask tracking",
    "pose": "Camera pose estimation",
    "point_track": "Point tracking",
    "pair_score": "Camera pair scoring",
    "solve": "Global offset solve",
}


class SyncProgress:
    steps = ("extract", "sync", "export")

    def __init__(self):
        self.restore()

    def restore(self, samples=None, analysis=None):
        self.steps_status = {}
        self.pipeline_window = None
        if samples is not None or analysis is not None:
            source = samples if samples is not None else analysis
            count = len(source["windows"]) * len(source["cameras"])
            self.set("extract", "complete", count, count)
        if analysis is not None:
            count = len(analysis["windows"])
            self.set("sync", "complete", count, count)

    def set(self, step, state, completed=0, total=None):
        self.steps_status[step] = {"state": state, "completed": completed, "total": total}

    def start(self, step):
        if step == "sync":
            self.reset_pipeline()
        self.set(step, "running")
        self.steps_status[step]["started_at"] = monotonic()

    def update(self, step, completed, total):
        self.steps_status[step].update(completed=completed, total=total)
        if step == "sync" and completed:
            self.stop_pipeline()

    def reset_pipeline(self):
        self.pipeline_window = None
        for stage in PIPELINE_STAGES:
            self.steps_status.pop(stage, None)

    def update_pipeline(self, window, windows, stage=None, completed=0, total=None):
        if stage is None:
            self.reset_pipeline()
            self.pipeline_window = (window, windows)
            return
        if completed == 0:
            self.start(stage)
        self.update(stage, completed, total)
        if total is not None and completed == total:
            self.finish(stage, True)

    def stop_pipeline(self):
        for stage in PIPELINE_STAGES:
            if self.steps_status.get(stage, {}).get("state") == "running":
                self.finish(stage, False)

    def snapshot(self):
        steps = {}
        for step, current in self.steps_status.items():
            steps[step] = status = current.copy()
            started_at = status.pop("started_at", None)
            if started_at is not None:
                status["elapsed_seconds"] = int(monotonic() - started_at)
        return steps

    def finish(self, step, success, *, preserve_sync=False):
        current = self.steps_status[step]
        current["elapsed_seconds"] = int(monotonic() - current.pop("started_at"))
        current["state"] = "complete" if success else "failed"
        if step == "sync":
            self.stop_pipeline()
        if success and step == "extract" and not preserve_sync:
            self.reset_pipeline()
        if success and step in self.steps:
            for later in self.steps[self.steps.index(step) + 1:]:
                if later != "sync" or not preserve_sync:
                    self.steps_status.pop(later, None)
