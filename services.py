"""Per-browser-session runtime state (robot engine, chat memory, temporary store).

Every visitor, guest or user, gets their own Runtime, so two people never share
a robot state. Guest runtimes expire after a period of inactivity.
"""

from __future__ import annotations

import re
import threading
import time

from database.session_store import TemporaryStore
from robot.decision_engine import DecisionEngine
from robot.robot_state import ChatMemory

DEFAULT_PREFS = {
    "ai_mode": "emotion",            # "normal" or "emotion"
    "tts_enabled": True,
    "voice_lang": "en-US",
    "ollama_model": "",              # empty = first installed model
    "min_confidence": 0.35,
    "camera_interval": 450,          # milliseconds between analysed frames
    "theme": "burgundy",
    "save_emotion_history": True,    # only affects logged-in users
    "save_chat_history": True,
    "save_preferences": True,
}
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9_.:/-]{0,80}$")


def clean_prefs(data: dict, base: dict) -> dict:
    """Keep only known settings with valid values."""
    prefs = dict(base)
    for key in ("tts_enabled", "save_emotion_history", "save_chat_history", "save_preferences"):
        if key in data:
            prefs[key] = bool(data[key])
    if data.get("ai_mode") in ("normal", "emotion"):
        prefs["ai_mode"] = data["ai_mode"]
    if data.get("theme") in ("burgundy", "noir"):
        prefs["theme"] = data["theme"]
    if data.get("voice_lang") in ("en-US", "en-IN", "en-GB", "hi-IN"):
        prefs["voice_lang"] = data["voice_lang"]
    if data.get("camera_interval") in (300, 450, 800):
        prefs["camera_interval"] = data["camera_interval"]
    if isinstance(data.get("min_confidence"), (int, float)):
        prefs["min_confidence"] = min(max(float(data["min_confidence"]), 0.2), 0.8)
    if isinstance(data.get("ollama_model"), str) and MODEL_NAME_RE.match(data["ollama_model"].strip()):
        prefs["ollama_model"] = data["ollama_model"].strip()
    return prefs


class Runtime:
    def __init__(self, sid: str, user_id: int | None = None, username: str | None = None) -> None:
        self.sid, self.user_id, self.username = sid, user_id, username
        self.engine = DecisionEngine()
        self.memory = ChatMemory()
        self.store = TemporaryStore()
        self.prefs = dict(DEFAULT_PREFS)
        self.db_session_id: int | None = None
        self.last_logged = float("-inf")
        self.last_detect = float("-inf")
        self.last_seen = time.monotonic()
        self.lock = threading.RLock()

    @property
    def is_guest(self) -> bool:
        return self.user_id is None

    def apply_prefs(self) -> None:
        self.engine.min_confidence = self.prefs["min_confidence"]

    def fresh_expression(self, max_age: float = 10.0):
        """The detected expression, but only if the camera saw a face very recently."""
        if self.engine.detected and time.monotonic() - self.last_detect <= max_age:
            return self.engine.chat_context()
        return None, None, None


class RuntimeRegistry:
    def __init__(self, guest_ttl: float = 7200, user_ttl: float = 43200) -> None:
        self._items: dict[str, Runtime] = {}
        self._lock = threading.Lock()
        self._ttl = {True: guest_ttl, False: user_ttl}
        self._pruned_at = time.monotonic()

    def add(self, runtime: Runtime) -> None:
        with self._lock:
            self._items[runtime.sid] = runtime

    def get(self, sid: str) -> Runtime | None:
        with self._lock:
            now = time.monotonic()
            if now - self._pruned_at > 60:
                self._pruned_at = now
                for key, item in list(self._items.items()):
                    if now - item.last_seen > self._ttl[item.is_guest]:
                        del self._items[key]
            runtime = self._items.get(sid)
            if runtime and now - runtime.last_seen > self._ttl[runtime.is_guest]:
                del self._items[sid]
                return None
            if runtime:
                runtime.last_seen = now
            return runtime

    def drop(self, sid: str) -> None:
        with self._lock:
            self._items.pop(sid, None)
