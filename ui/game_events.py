from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional
import time


@dataclass
class GameEventStore:
    """In-memory event/state store for browser UI and future experiment logging."""

    events: List[Dict[str, Any]] = field(default_factory=list)
    snapshot: Dict[str, Any] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def reset(self):
        with self._lock:
            self.events = []
            self.snapshot = {
                "status": "idle",
                "turn": 0,
                "round": 1,
                "phase": "introduction",
                "current_speaker": None,
                "agent_thoughts": {},
                "history": [],
                "accusations": {},
                "agent_memory": {},
                "murderer": None,
                "verdict": None,
                "scenario_title": "The Business of Murder",
                "scenario_location": None,
                "started_at": None,
                "updated_at": time.time(),
                "error": None,
            }

    def append(self, event_type: str, payload: Optional[Dict[str, Any]] = None):
        payload = payload or {}
        event = {
            "index": len(self.events),
            "type": event_type,
            "timestamp": time.time(),
            "payload": payload,
        }
        with self._lock:
            self.events.append(event)
            self._apply_event(event_type, payload)
            self.snapshot["updated_at"] = event["timestamp"]
        return event

    def _apply_event(self, event_type: str, payload: Dict[str, Any]):
        if event_type == "run_started":
            self.snapshot["scenario_title"] = payload.get("scenario_title", self.snapshot.get("scenario_title"))
            self.snapshot["scenario_location"] = payload.get("scenario_location", self.snapshot.get("scenario_location"))
        elif event_type == "game_started":
            self.snapshot.update({
                "status": "running",
                "started_at": payload.get("started_at", time.time()),
                "round": payload.get("round", 1),
                "phase": payload.get("phase", "introduction"),
                "turn": payload.get("turn", 0),
                "murderer": payload.get("murderer"),
                "scenario_title": payload.get("scenario_title", self.snapshot.get("scenario_title")),
                "scenario_location": payload.get("scenario_location", self.snapshot.get("scenario_location")),
                "history": [],
                "accusations": {},
                "verdict": None,
                "error": None,
            })
        elif event_type == "turn_started":
            self.snapshot["turn"] = payload.get("turn", self.snapshot.get("turn", 0))
            self.snapshot["round"] = payload.get("round", self.snapshot.get("round", 1))
            self.snapshot["phase"] = payload.get("phase", self.snapshot.get("phase", "introduction"))
        elif event_type == "thoughts_generated":
            self.snapshot["agent_thoughts"] = payload.get("thoughts", {})
        elif event_type == "speaker_selected":
            self.snapshot["current_speaker"] = payload.get("speaker")
        elif event_type == "utterance":
            utterance = payload.get("utterance")
            if utterance:
                self.snapshot.setdefault("history", []).append(utterance)
                self.snapshot["current_speaker"] = utterance.get("speaker")
        elif event_type == "round_changed":
            self.snapshot["round"] = payload.get("round", self.snapshot.get("round", 1))
            self.snapshot["phase"] = payload.get("phase", self.snapshot.get("phase", "discussion"))
        elif event_type == "accusation":
            agent = payload.get("agent")
            result = payload.get("result")
            if agent and result:
                self.snapshot.setdefault("accusations", {})[agent] = result
        elif event_type == "memory_updated":
            memory = payload.get("agent_memory", {})
            if memory:
                self.snapshot["agent_memory"] = memory
        elif event_type == "game_finished":
            self.snapshot["status"] = "finished"
            self.snapshot["verdict"] = payload.get("verdict")
            self.snapshot["murderer"] = payload.get("murderer", self.snapshot.get("murderer"))
        elif event_type == "game_error":
            self.snapshot["status"] = "error"
            self.snapshot["error"] = payload.get("error")

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "snapshot": dict(self.snapshot),
                "events": list(self.events),
            }


STORE = GameEventStore()
STORE.reset()
