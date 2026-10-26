"""Flask application: Emotion-Aware Intelligent Robot.

Pipeline: webcam frame -> face detection -> emotion classification -> decision
engine -> robot state -> (Ollama chat) -> voice/text -> storage -> analytics.

Guests keep everything in a temporary in-memory store. Logged-in users also save
to SQLite, always filtered by their own user_id.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
from datetime import datetime
from functools import wraps
from pathlib import Path
from time import monotonic, perf_counter

from flask import (Flask, Response, flash, g, jsonify, redirect, render_template, request, session, url_for)
from werkzeug.exceptions import HTTPException

import auth
from ai.ollama_client import OllamaClient
from database import analytics
from database.db import Database, DatabaseError, DuplicateUserError
from robot.speech import TextToSpeech
from services import DEFAULT_PREFS, Runtime, RuntimeRegistry, clean_prefs
from vision import preprocessing
from vision.emotion_classifier import EmotionClassifier
from vision.errors import InvalidImageError, VisionError
from vision.face_detector import FaceDetector

PROJECT_ROOT = Path(__file__).resolve().parent
MAX_FRAME_BYTES = 2_000_000
MAX_MESSAGE_LENGTH = 500
MAX_MEMORY_LENGTH = 200
LOG_INTERVAL_SECONDS = 2.0
REMEMBER_RE = re.compile(r"^\s*(?:please\s+)?remember\s+(?:that\s+)?(.{3,200}?)\s*[.!]?\s*$", re.I)
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _secret_key(data_dir: Path) -> str:
    path = data_dir / "secret_key"
    if not path.exists():
        path.write_text(secrets.token_hex(32))
    return path.read_text().strip()


def create_app(face_detector=None, emotion_classifier=None, ollama=None, tts=None, database=None,
               secret_key=None, guest_ttl=7200, data_dir=None) -> Flask:
    app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"), static_folder=str(PROJECT_ROOT / "static"))
    data_dir = Path(data_dir or PROJECT_ROOT / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    app.config.update(SECRET_KEY=secret_key or _secret_key(data_dir), SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE="Lax", MAX_CONTENT_LENGTH=MAX_FRAME_BYTES + 100_000)

    problems: dict[str, str] = {}

    def build(name, factory, given):
        if given is not None:
            return given
        try:
            return factory()
        except (VisionError, DatabaseError) as error:
            problems[name] = str(error)
            app.logger.error("%s unavailable: %s", name, error)
            return None

    detector = build("face_detector", FaceDetector, face_detector)
    classifier = build("emotion_model", EmotionClassifier, emotion_classifier)
    db = build("database", lambda: Database(data_dir / "robot.db"), database)
    ollama = ollama or OllamaClient()
    speech = tts or TextToSpeech()
    registry = RuntimeRegistry(guest_ttl=guest_ttl)
    detect_lock = threading.Lock()

    # ------------------------------------------------------------ helpers
    def db_call(method, *args, default=None, **kwargs):
        """Run a database method; on failure keep the app alive and remember the problem."""
        if db is None:
            return default
        try:
            return getattr(db, method)(*args, **kwargs)
        except DatabaseError as error:
            problems["database"] = str(error)
            app.logger.error("Database error: %s", error)
            return default

    def fail(message, code, status):
        return jsonify({"error": message, "code": code}), status

    def new_runtime(user=None) -> Runtime:
        runtime = Runtime(secrets.token_hex(16), user["id"] if user else None, user["username"] if user else None)
        if user:
            runtime.prefs = clean_prefs(db_call("get_prefs", user["id"], default={}), DEFAULT_PREFS)
            runtime.db_session_id = db_call("start_session", user["id"])
        runtime.apply_prefs()
        registry.add(runtime)
        session["sid"] = runtime.sid
        return runtime

    def needs_session(api=False):
        def decorator(view):
            @wraps(view)
            def wrapper(*args, **kwargs):
                if g.rt is None:
                    if api:
                        if g.expired:
                            return fail("Your guest session expired and its temporary data was cleared.", "session_expired", 401)
                        return fail("Please log in or continue as a guest first.", "login_required", 401)
                    return redirect(url_for("landing"))
                return view(*args, **kwargs)
            return wrapper
        return decorator

    def needs_user(view):
        @wraps(view)
        def wrapper(*args, **kwargs):
            if g.rt is None or g.rt.is_guest:
                return fail("This needs a logged-in account. Guest data is temporary.", "login_required", 403)
            return view(*args, **kwargs)
        return wrapper

    # ------------------------------------------------------------ request lifecycle
    @app.before_request
    def load_session():
        g.rt, g.expired = None, False
        if request.endpoint == "static":
            return None
        sid = session.get("sid")
        if sid:
            g.rt = registry.get(sid)
            if g.rt is None:
                if session.get("mode") == "user" and db is not None:
                    user = db_call("get_user", session.get("user_id"))
                    if user:
                        g.rt = new_runtime(user)
                    else:
                        session.clear()
                else:
                    session.clear()
                    session["expired"] = True
                    g.expired = True
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            token = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token") or ""
            if not secrets.compare_digest(token, session["csrf"]):
                return fail("Your page is out of date. Please refresh and try again.", "csrf", 400)
        return None

    @app.context_processor
    def inject():
        rt = g.get("rt")
        info = {"guest": rt.is_guest if rt else True, "username": rt.username if rt else None,
                "prefs": rt.prefs if rt else DEFAULT_PREFS, "csrf": session.get("csrf", "")}
        return {"rt": rt, "is_guest": info["guest"], "username": info["username"], "prefs": info["prefs"],
                "csrf_token": info["csrf"], "app_info": info}

    @app.errorhandler(413)
    def too_large(_):
        return fail("The camera frame is too large. Lower the camera resolution.", "frame_too_large", 413)

    @app.errorhandler(Exception)
    def unexpected(exc):
        if isinstance(exc, HTTPException):
            return jsonify({"error": exc.description, "code": "http_error"}), exc.code
        app.logger.exception("Unhandled error")
        return fail("Something went wrong on the server.", "internal_error", 500)

    # ------------------------------------------------------------ landing, login, guest
    @app.get("/")
    def landing():
        if g.rt:
            return redirect(url_for("dashboard"))
        if session.pop("expired", None):
            flash("Your guest session expired, so its temporary data was cleared.")
        return render_template("landing.html")

    @app.post("/guest")
    def guest():
        session["mode"] = "guest"
        new_runtime()
        return redirect(url_for("dashboard"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "GET":
            return render_template("register.html", errors={}, form={})
        form = {k: request.form.get(k, "").strip() for k in ("username", "email")}
        password, confirm = request.form.get("password", ""), request.form.get("confirm", "")
        errors = auth.validate_registration(form["username"], form["email"], password, confirm)
        if db is None:
            errors["form"] = "Accounts are unavailable right now (database problem). You can continue as a guest."
        if not errors:
            try:
                user_id = db.create_user(form["username"], form["email"], auth.hash_password(password))
            except DuplicateUserError as error:
                errors[error.field] = str(error)
            except DatabaseError:
                errors["form"] = "Could not create the account because of a database problem."
        if errors:
            return render_template("register.html", errors=errors, form=form), 400
        return start_user_session(db_call("get_user", user_id))

    def start_user_session(user):
        old_csrf = secrets.token_hex(16)
        session.clear()
        session.update(mode="user", user_id=user["id"], csrf=old_csrf)
        new_runtime(user)
        return redirect(url_for("dashboard"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "GET":
            return render_template("login.html", error=None)
        user = db_call("find_user", request.form.get("identifier", "").strip()) if db else None
        if db is None:
            return render_template("login.html", error="Accounts are unavailable right now. You can continue as a guest."), 503
        if not user or not auth.verify_password(user["password_hash"], request.form.get("password", "")):
            return render_template("login.html", error="Invalid username or password."), 401
        return start_user_session(user)

    @app.post("/logout")
    def logout():
        rt = g.rt
        if rt:
            if rt.user_id and rt.db_session_id:
                db_call("end_session", rt.user_id, rt.db_session_id)
            registry.drop(rt.sid)
            flash("Guest session ended. Temporary data was cleared." if rt.is_guest else "You have been logged out.")
        session.clear()
        return redirect(url_for("landing"))

    # ------------------------------------------------------------ pages
    def page(name, template):
        @needs_session()
        def view():
            return render_template(template, page=name)
        view.__name__ = name
        app.add_url_rule(f"/{name}", name, view)

    for route, template in (("dashboard", "dashboard.html"), ("emotion", "emotion.html"), ("graphs", "graphs.html"),
                            ("statistics", "statistics.html"), ("reports", "reports.html"), ("memory", "memory.html"),
                            ("profile", "profile.html"), ("settings", "settings.html")):
        page(route, template)

    @app.get("/how-it-works")
    def how_it_works():
        return render_template("how.html", page="how")

    @app.get("/chat")
    @needs_session()
    def chat_page():
        return render_template("chat.html", page="chat")

    # ------------------------------------------------------------ health and status
    @app.get("/health")
    def health():
        models_ok = detector is not None and classifier is not None
        return jsonify({"status": "ok" if models_ok else "degraded", "face_detector": detector is not None,
                        "emotion_model": classifier is not None, "database": db is not None and "database" not in problems,
                        "voice_available": speech.available, "ollama": ollama.status(), "problems": problems})

    @app.get("/status")
    @needs_session(api=True)
    def status():
        rt = g.rt
        with rt.lock:
            emotion, confidence, _ = rt.fresh_expression()
            summary = analytics.summarise(rt.store.detections(), rt.store.interactions())
            return jsonify({"robot": rt.engine.state.as_dict(),
                            "expression": {"emotion": emotion, "confidence": confidence} if emotion else None,
                            "session": summary, "ollama": ollama.status(rt.prefs["ollama_model"] or None),
                            "problems": problems})

    # ------------------------------------------------------------ vision
    @app.post("/detect")
    @needs_session(api=True)
    def detect():
        rt = g.rt
        if detector is None or classifier is None:
            return fail(problems.get("emotion_model") or problems.get("face_detector") or "The vision models are not loaded.",
                        "model_missing", 503)
        data = request.get_data(cache=False, as_text=False)
        if len(data) > MAX_FRAME_BYTES:
            return fail("The camera frame is too large. Lower the camera resolution.", "frame_too_large", 413)
        try:
            frame = preprocessing.decode_image(data)
            started = perf_counter()
            faces = []
            with detect_lock:
                for box in detector.detect(frame):
                    crop = preprocessing.crop_face(frame, box.x, box.y, box.width, box.height)
                    if crop is None:
                        continue
                    prediction = classifier.predict(crop)
                    faces.append({**box.as_dict(), "face_confidence": box.confidence, "emotion": prediction.emotion,
                                  "confidence": prediction.confidence, "probabilities": prediction.probabilities})
            inference_ms = round((perf_counter() - started) * 1000, 1)
        except InvalidImageError as exc:
            return fail(str(exc), "invalid_frame", 400)
        except VisionError as exc:
            app.logger.exception("Vision pipeline failed")
            return fail(str(exc), "analysis_failed", 500)

        notice, primary_index, robot, logged = None, None, None, False
        with rt.lock:
            rt.apply_prefs()
            if not faces:
                rt.engine.no_face()
                notice = "no_face"
            else:
                rt.last_detect = monotonic()
                if len(faces) > 1:
                    notice = "multiple_faces"
                primary_index = max(range(len(faces)), key=lambda i: faces[i]["width"] * faces[i]["height"])
                primary = faces[primary_index]
                decision = rt.engine.observe(primary["probabilities"])
                primary.update(emotion=decision.detected, confidence=decision.confidence,
                               probabilities=decision.probabilities, uncertain=decision.uncertain)
                if decision.uncertain and notice is None:
                    notice = "low_confidence"
                if not decision.uncertain and monotonic() - rt.last_logged >= LOG_INTERVAL_SECONDS:
                    rt.last_logged = monotonic()
                    rt.store.add_detection(decision.detected, decision.confidence, decision.probabilities,
                                           rt.engine.state.status, decision.action)
                    if rt.user_id and rt.prefs["save_emotion_history"]:
                        db_call("add_detection", rt.user_id, rt.db_session_id, decision.detected, decision.confidence,
                                decision.probabilities, rt.engine.state.status, decision.action)
                    logged = True
                voice_queued = bool(decision.speak and rt.prefs["tts_enabled"] and speech.speak(decision.message))
                robot = {"emotion": decision.emotion, "mood": decision.mood, "action": decision.action,
                         "style": decision.style, "status": rt.engine.state.status, "message": decision.message,
                         "changed": decision.changed, "joke_offer": decision.joke_offer, "voice_queued": voice_queued}
        return jsonify({"frame": {"width": frame.shape[1], "height": frame.shape[0]}, "faces": faces,
                        "face_count": len(faces), "primary_index": primary_index, "notice": notice, "robot": robot,
                        "logged": logged, "time": datetime.now().strftime("%H:%M:%S"), "inference_ms": inference_ms})

    # ------------------------------------------------------------ memory (explicit preferences only)
    def memory_items(rt):
        items = [{"id": f"t-{m['id']}", "text": m["text"], "created_at": m["created_at"], "persistent": False}
                 for m in rt.store.memories()]
        if rt.user_id:
            saved = db_call("memories", rt.user_id, default=[])
            items = [{"id": f"p-{m['id']}", "text": m["text"], "created_at": m["created_at"], "persistent": True}
                     for m in saved] + items
        return items

    def add_memory(rt, text):
        text = CONTROL_RE.sub(" ", text).strip()[:MAX_MEMORY_LENGTH]
        if len(text) < 3:
            return None
        if rt.user_id and rt.prefs["save_preferences"]:
            if db_call("add_memory", rt.user_id, text) is not None:
                return True
        rt.store.add_memory(text)
        return False  # temporary only

    @app.get("/api/memory")
    @needs_session(api=True)
    def memory_list():
        return jsonify({"items": memory_items(g.rt), "guest": g.rt.is_guest})

    @app.post("/api/memory")
    @needs_session(api=True)
    def memory_add():
        text = str((request.get_json(silent=True) or {}).get("text", ""))
        saved = add_memory(g.rt, text)
        if saved is None:
            return fail("Please write a short preference (at least 3 characters).", "empty_memory", 400)
        return jsonify({"items": memory_items(g.rt), "persistent": saved})

    @app.delete("/api/memory/<memory_id>")
    @needs_session(api=True)
    def memory_delete(memory_id):
        rt = g.rt
        kind, _, number = memory_id.partition("-")
        if not number.isdigit() or kind not in ("t", "p"):
            return fail("Unknown memory.", "bad_memory", 400)
        if kind == "p":
            removed = bool(rt.user_id and db_call("delete_memory", rt.user_id, int(number), default=False))
        else:
            removed = rt.store.delete_memory(int(number))
        return jsonify({"removed": removed, "items": memory_items(rt)})

    @app.delete("/api/memory")
    @needs_session(api=True)
    def memory_clear():
        rt = g.rt
        rt.store.clear_memories()
        if rt.user_id:
            db_call("clear_memories", rt.user_id)
        return jsonify({"items": memory_items(rt)})

    # ------------------------------------------------------------ chat
    @app.post("/chat")
    @needs_session(api=True)
    def chat():
        rt = g.rt
        payload = request.get_json(silent=True) or {}
        message = CONTROL_RE.sub(" ", str(payload.get("message", ""))).strip()[:MAX_MESSAGE_LENGTH]
        if not message:
            return fail("Please type or say something first.", "empty_message", 400)
        input_type = "voice" if payload.get("input_type") == "voice" else "text"
        mode = payload.get("mode") if payload.get("mode") in ("normal", "emotion") else rt.prefs["ai_mode"]

        with rt.lock:
            rt.engine.state.update(status="Thinking")
            rt.memory.add("user", message)
            rt.engine.dismiss_joke_offer()
            expression, confidence, style = rt.fresh_expression()
            history, memories = rt.memory.messages(), [m["text"] for m in memory_items(rt)]

        reply, notice, llm_used = None, None, False
        remember = REMEMBER_RE.match(message)
        if remember:
            saved = add_memory(rt, remember.group(1))
            reply = ("I'll remember that for future conversations." if saved
                     else "I'll remember that for this session only." if saved is False
                     else "Tell me what to remember, for example: remember that I prefer short answers.")
        else:
            model = rt.prefs["ollama_model"] or None
            reply = ollama.reply(history, expression, confidence, style, mode, memories, model)  # slow: outside the lock
            llm_used = reply is not None
            if reply is None:
                notice = ollama.status(model)["message"]
                with rt.lock:
                    reply = rt.engine.offline_replies.reply(message, expression, confidence)

        with rt.lock:
            rt.memory.add("assistant", reply)
            rt.engine.state.update(status="Responding", response=reply)
            robot = rt.engine.state.as_dict()
            rt.store.add_interaction(message, reply, input_type, expression, confidence, robot["action"], llm_used)
            if rt.user_id and rt.prefs["save_chat_history"]:
                db_call("add_interaction", rt.user_id, rt.db_session_id, message, reply, input_type, expression,
                        confidence, robot["action"], llm_used)
        voice_queued = bool(rt.prefs["tts_enabled"] and speech.speak(reply))
        return jsonify({"reply": reply, "llm_used": llm_used, "notice": notice, "input_type": input_type, "mode": mode,
                        "expression": expression, "voice_queued": voice_queued, "robot": robot})

    # ------------------------------------------------------------ data: history, stats, graphs
    def data_for(rt, scope):
        """Newest-first detections and interactions for the requested scope."""
        if scope == "history" and rt.user_id and db:
            return (db_call("detections", rt.user_id, default=[]), db_call("interactions", rt.user_id, default=[]),
                    db_call("session_count", rt.user_id, default=1), "history")
        return rt.store.detections(), rt.store.interactions(), 1, "session"

    @app.get("/history")
    @needs_session(api=True)
    def history():
        limit = max(1, min(request.args.get("limit", 60, type=int), 500))
        dets, ints, _, scope = data_for(g.rt, request.args.get("scope", "session"))
        kind = "interactions" if request.args.get("kind") == "interactions" else "detections"
        return jsonify({"kind": kind, "scope": scope, "items": (ints if kind == "interactions" else dets)[:limit]})

    @app.get("/stats")
    @needs_session(api=True)
    def stats():
        dets, ints, sessions, scope = data_for(g.rt, request.args.get("scope", "session"))
        result = analytics.summarise(dets, ints, sessions)
        result.update(scope=scope, started_at=g.rt.store.started_at if scope == "session" else None)
        return jsonify(result)

    @app.get("/api/graphs")
    @needs_session(api=True)
    def graphs_api():
        rt = g.rt
        dets, ints, sessions, scope = data_for(rt, request.args.get("scope", "session"))
        points = [{"ts": d["ts"], "emotion": d["emotion"], "confidence": d["confidence"]} for d in dets[:120][::-1]]
        comparison = None
        if scope == "history":
            rows = db_call("session_summaries", rt.user_id, default=[])
            if len(rows) >= 2:
                comparison = [{"session_id": r["session_id"], "started_at": r["started_at"], "detections": r["detections"],
                               "average_confidence": round(r["average_confidence"], 4)} for r in rows[-10:]]
        return jsonify({"scope": scope, "distribution": analytics.summarise(dets, ints, sessions)["distribution"],
                        "points": points, "comparison": comparison})

    # ------------------------------------------------------------ reports
    def build_report(rt):
        dets, ints = rt.store.detections(), rt.store.interactions()
        return {"project": "Emotion-Aware Intelligent Robot", "report_type": "current_session",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "session": {"type": "guest (temporary)" if rt.is_guest else "account", "username": rt.username,
                            "started_at": rt.store.started_at},
                "statistics": analytics.summarise(dets, ints, 1),
                "timeline_summary": analytics.timeline_summary(dets[::-1]),
                "notes": ["Expressions are classifications of visible facial expressions, not diagnoses and not proof of real feelings."]}

    def json_download(content, name):
        return Response(json.dumps(content, indent=2), mimetype="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/report")
    @needs_session(api=True)
    def report():
        return json_download(build_report(g.rt), "emotion_robot_session_report.json")

    @app.route("/api/reports", methods=["GET", "POST"])
    @needs_session(api=True)
    @needs_user
    def reports_api():
        rt = g.rt
        if request.method == "POST":
            content = build_report(rt)
            content["session"]["saved_to_account"] = True
            report_id = db_call("save_report", rt.user_id, rt.db_session_id, f"Session report {content['generated_at']}", content)
            if report_id is None:
                return fail("Could not save the report (database problem).", "report_failed", 500)
        return jsonify({"items": db_call("list_reports", rt.user_id, default=[])})

    @app.get("/reports/<int:report_id>/download")
    @needs_session(api=True)
    @needs_user
    def report_download(report_id):
        row = db_call("get_report", g.rt.user_id, report_id)
        if not row:
            return fail("Report not found.", "not_found", 404)
        return json_download(row["content"], f"emotion_robot_report_{report_id}.json")

    # ------------------------------------------------------------ settings, profile, account
    @app.route("/api/settings", methods=["GET", "POST"])
    @needs_session(api=True)
    def settings_api():
        rt = g.rt
        if request.method == "POST":
            rt.prefs = clean_prefs(request.get_json(silent=True) or {}, rt.prefs)
            rt.apply_prefs()
            if rt.user_id:
                db_call("set_prefs", rt.user_id, rt.prefs)
        return jsonify({"prefs": rt.prefs, "guest": rt.is_guest})

    @app.get("/api/ollama/models")
    @needs_session(api=True)
    def ollama_models():
        return jsonify({"models": ollama.installed(), "status": ollama.status(g.rt.prefs["ollama_model"] or None)})

    @app.get("/api/profile")
    @needs_session(api=True)
    @needs_user
    def profile_api():
        rt = g.rt
        user = db_call("get_user", rt.user_id)
        return jsonify({"username": user["username"], "email": user["email"], "created_at": user["created_at"],
                        "counts": db_call("counts", rt.user_id, default={})})

    @app.post("/api/profile/password")
    @needs_session(api=True)
    @needs_user
    def change_password():
        payload = request.get_json(silent=True) or {}
        user = db_call("get_user", g.rt.user_id)
        if not auth.verify_password(user["password_hash"], str(payload.get("current", ""))):
            return fail("Your current password is not correct.", "bad_password", 403)
        new = str(payload.get("new", ""))
        if len(new) < 8:
            return fail("The new password must be at least 8 characters.", "weak_password", 400)
        db_call("update_password", g.rt.user_id, auth.hash_password(new))
        return jsonify({"ok": True})

    @app.post("/api/account/delete-data")
    @needs_session(api=True)
    @needs_user
    def delete_data():
        rt = g.rt
        db_call("delete_user_data", rt.user_id)
        with rt.lock:
            rt.store.clear_history()
            rt.store.clear_memories()
            rt.engine.reset()
            rt.memory.clear()
            rt.prefs = dict(DEFAULT_PREFS)
        return jsonify({"ok": True})

    @app.post("/api/account/delete")
    @needs_session(api=True)
    @needs_user
    def delete_account():
        rt = g.rt
        user = db_call("get_user", rt.user_id)
        if not auth.verify_password(user["password_hash"], str((request.get_json(silent=True) or {}).get("password", ""))):
            return fail("Password is not correct.", "bad_password", 403)
        db_call("delete_user", rt.user_id)
        registry.drop(rt.sid)
        session.clear()
        return jsonify({"ok": True})

    @app.post("/session/reset")
    @needs_session(api=True)
    def reset():
        """Clear the CURRENT session only (saved history from earlier sessions stays)."""
        rt = g.rt
        with rt.lock:
            rt.engine.reset()
            rt.memory.clear()
            rt.store.clear_history()
            rt.last_logged = float("-inf")
            rt.apply_prefs()
        if rt.user_id and rt.db_session_id:
            db_call("delete_session_records", rt.user_id, rt.db_session_id)
        return jsonify({"ok": True})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")), debug=False, threaded=True)
