from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import json
import subprocess


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_git_commit(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


class EventLogger:
    def __init__(self, run_dir: Path, manifest: Dict[str, Any]):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.index = 0
        self.manifest: Dict[str, Any] = dict(manifest)
        self.manifest.setdefault("started_at", utc_now_iso())
        self.manifest.setdefault("status", "running")
        self.write_manifest()

    def write_manifest(self):
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, indent=2, sort_keys=True)

    def append(self, event_type: str, payload: Optional[Dict[str, Any]] = None):
        event = {
            "index": self.index,
            "timestamp": utc_now_iso(),
            "type": event_type,
            "run_id": self.manifest.get("run_id"),
            "payload": payload or {},
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.index += 1
        return event

    def finalize(self, status: str = "finished", extra: Optional[Dict[str, Any]] = None):
        self.manifest["status"] = status
        self.manifest["ended_at"] = utc_now_iso()
        if extra:
            self.manifest.update(extra)
        self.write_manifest()


class MultiEventSink:
    def __init__(self, sinks: Iterable[Any]):
        self.sinks = [sink for sink in sinks if sink is not None]

    def append(self, event_type: str, payload: Optional[Dict[str, Any]] = None):
        result = None
        for sink in self.sinks:
            result = sink.append(event_type, payload or {})
        return result
