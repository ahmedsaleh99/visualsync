"""Persist and validate completed synchronization stages for one window."""

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


STAGES = ("segment", "mask_track", "pose", "point_track", "pair_score", "solve")


def file_identity(paths):
    result = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        result.update(json.dumps([str(path), stat.st_size, stat.st_mtime_ns]).encode())
    return result.hexdigest()


class PipelineCheckpoints:
    def __init__(self, output, frames, reuse):
        self.path = output / "stage_checkpoints.json"
        self.output = output
        self.reuse = reuse
        self.inputs = file_identity(frames)
        self.entries = {}
        try:
            saved = json.loads(self.path.read_text())
            if (saved.get("version") == 1 and saved.get("inputs") == self.inputs
                    and isinstance(saved.get("stages"), dict)):
                self.entries = saved["stages"]
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            pass

    def previous(self, stage, settings):
        entry = self.entries.get(stage)
        if self.reuse and isinstance(entry, dict) and entry.get("settings") == settings:
            try:
                if file_identity(self.output / name for name in entry["files"]) == entry["identity"]:
                    return entry
            except (OSError, TypeError, KeyError):
                pass
        for later in STAGES[STAGES.index(stage):]:
            self.entries.pop(later, None)
        self.write()
        return None

    def save(self, stage, settings, files, **data):
        files = list(files)
        if not all(path.is_file() and path.stat().st_size > 0 for path in files):
            return
        self.entries[stage] = {
            "settings": settings,
            "files": [str(path.relative_to(self.output)) for path in files],
            "identity": file_identity(files),
            **data,
        }
        self.write()

    def write(self):
        with NamedTemporaryFile(mode="w", dir=self.output, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump({"version": 1, "inputs": self.inputs, "stages": self.entries}, stream)
                stream.close()
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)
