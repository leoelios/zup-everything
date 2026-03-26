"""
Zup Web — orchestrator dashboard backend.
Manages multiple agent sessions and streams events to the browser via WebSockets.
"""

import asyncio
import json
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make zup-cli modules importable
sys.path.insert(0, str(Path(__file__).parent.parent / "zup-cli"))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Zup Agent Orchestrator")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

class _Interruptible:
    """A threading.Event wrapper that yields a value when resolved."""
    def __init__(self):
        self._evt = threading.Event()
        self._value = None

    def resolve(self, value):
        self._value = value
        self._evt.set()

    def wait(self, timeout=300):
        self._evt.wait(timeout=timeout)
        return self._value


sessions: dict[str, dict] = {}          # session_id -> session dict
event_queues: dict[str, asyncio.Queue] = {}   # session_id -> asyncio.Queue

_HISTORY_TYPES = frozenset({
    "user_message", "user_message_queued", "thinking",
    "tool_use", "tool_result", "response", "info", "error",
    "ask_user", "confirm_request", "bash_line",
})


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _evt(type: str, **data) -> dict:
    return {"type": type, "ts": _now(), **data}


# ---------------------------------------------------------------------------
# Streaming thinking filter
# ---------------------------------------------------------------------------

class _TagStripper:
    """
    Strips a specific XML tag (open…close) from streaming chunks.
    Content inside the tag is discarded; clean text is forwarded to on_clean_chunk.
    Handles tags split across chunk boundaries via internal buffering.
    """

    def __init__(self, open_tag: str, close_tag: str, on_clean_chunk):
        self._open    = open_tag
        self._close   = close_tag
        self._on_chunk = on_clean_chunk
        self._buf     = ""
        self._inside  = False

    def feed(self, chunk: str):
        self._buf += chunk
        while True:
            if not self._inside:
                pos = self._buf.find(self._open)
                if pos == -1:
                    safe_end = max(0, len(self._buf) - len(self._open) + 1)
                    if safe_end:
                        self._on_chunk(self._buf[:safe_end])
                        self._buf = self._buf[safe_end:]
                    break
                if pos > 0:
                    self._on_chunk(self._buf[:pos])
                self._buf    = self._buf[pos + len(self._open):]
                self._inside = True
            else:
                pos = self._buf.find(self._close)
                if pos == -1:
                    safe_end = max(0, len(self._buf) - len(self._close) + 1)
                    self._buf = self._buf[safe_end:]
                    break
                self._buf    = self._buf[pos + len(self._close):]
                self._inside = False


# Keep old name as alias for callers
_ThinkingFilter = lambda on_clean_chunk: _TagStripper("<thinking>", "</thinking>", on_clean_chunk)


# ---------------------------------------------------------------------------
# Agent runner (background thread)
# ---------------------------------------------------------------------------

def _patch_display(put) -> dict:
    """
    Redirect display module output to the web event queue for this session.
    Uses thread-local overrides so parallel sessions remain fully isolated.
    Returns a dict of previous thread-local values to restore afterward.
    """
    import display as _display

    _keys = [
        "print_info", "print_thinking", "print_tool_use", "print_tool_result",
        "print_response", "print_separator", "stream_start", "stream_chunk",
        "stream_stop", "spinner_start", "spinner_stop", "bash_output",
    ]
    # Save whatever was previously stored in this thread's local (None = not set)
    saved = {k: getattr(_display._thread_local, k, None) for k in _keys}

    _disp_tool_strip  = _TagStripper("<tool_call>", "</tool_call>",
                                     on_clean_chunk=lambda text: put(_evt("chunk", text=text)))
    _disp_think_strip = _TagStripper("<thinking>", "</thinking>",
                                     on_clean_chunk=_disp_tool_strip.feed)

    _display._thread_local.print_info        = lambda msg: put(_evt("info", text=str(msg)))
    _display._thread_local.print_thinking    = lambda text: put(_evt("thinking", text=str(text)))
    _display._thread_local.print_tool_use    = lambda name, params: put(_evt("tool_use", name=name, params=params))
    _display._thread_local.print_tool_result = lambda name, result: put(_evt("tool_result", name=name, result=str(result)[:800]))
    _display._thread_local.print_response    = lambda text: put(_evt("response", text=str(text))) if text and str(text).strip() else None
    _display._thread_local.print_separator   = lambda: None
    _display._thread_local.stream_start      = lambda in_chars=0: put(_evt("llm_start", prompt_len=in_chars))
    _display._thread_local.stream_chunk      = lambda text: _disp_think_strip.feed(text)
    _display._thread_local.stream_stop       = lambda: None
    _display._thread_local.spinner_start     = lambda label="", status="": None
    _display._thread_local.spinner_stop      = lambda: None
    _display._thread_local.bash_output       = lambda line, is_stderr=False: put(_evt("bash_line", line=line, is_stderr=is_stderr))

    return saved


def _restore_display(saved: dict):
    import display as _display
    for attr, fn in saved.items():
        if fn is None:
            try:
                delattr(_display._thread_local, attr)
            except AttributeError:
                pass
        else:
            setattr(_display._thread_local, attr, fn)


def _run_agent(session_id: str, message: str, loop: asyncio.AbstractEventLoop):
    """Execute one agent turn in a background thread, emitting events into the queue."""
    import agent as agent_module
    from modifiers import extract_modifiers, apply_modifiers, PASSTHROUGH_MODIFIERS

    session = sessions[session_id]
    queue = event_queues[session_id]

    def put(event: dict):
        if event.get("type") in _HISTORY_TYPES:
            session["history"].append(event)
        asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    ag = session["agent"]

    # Redirect display module to the web event queue (affects modifiers too)
    _saved_display = _patch_display(put)

    # Chain strippers: tool_call tags → thinking tags → browser chunk events
    _tool_stripper  = _TagStripper("<tool_call>", "</tool_call>",
                                   on_clean_chunk=lambda text: put(_evt("chunk", text=text)))
    _think_stripper = _TagStripper("<thinking>",  "</thinking>",
                                   on_clean_chunk=_tool_stripper.feed)
    def _on_chunk(text):
        if session.get("stop_requested"):
            raise InterruptedError("Agent stopped by user")
        _think_stripper.feed(text)
    ag.on_llm_chunk    = _on_chunk
    ag.on_thinking     = lambda text: put(_evt("thinking", text=text))
    ag.on_tool_use     = lambda name, params: put(_evt("tool_use", name=name, params=params))
    ag.on_tool_result  = lambda name, result: put(_evt("tool_result", name=name, result=result[:800]))
    ag.on_llm_activity = lambda hint: put(_evt("activity", hint=hint))
    ag.on_llm_start    = lambda n: put(_evt("llm_start", prompt_len=n))
    ag.on_token_count  = lambda i, o: put(_evt("tokens", input=i, output=o))

    # Tool confirmation — suspend the agent thread until the UI responds
    def on_confirm(name: str, params: dict) -> bool:
        req = _Interruptible()
        session["pending_confirm"] = req
        put(_evt("confirm_request", name=name, params=params))
        result = req.wait(timeout=120)  # 2-min timeout defaults to allow
        session["pending_confirm"] = None
        return True if result is None else result

    ag.on_confirm_tool = on_confirm

    # Override ask_user so it routes through the web UI
    def web_ask_user(question: str, options=None):
        req = _Interruptible()
        session["pending_ask_user"] = req
        put(_evt("ask_user", question=question, options=options or []))
        answer = req.wait(timeout=300) or ""
        session["pending_ask_user"] = None
        return answer

    ag._tool_registry["ask_user"] = web_ask_user

    # ── Extract @modifiers ────────────────────────────────────────────────────
    mods, clean_message = extract_modifiers(message)

    # @insecure — auto-confirm all tools without prompting
    if "insecure" in mods:
        ag.on_confirm_tool = lambda n, p: True

    # Set working directory for this session (web default: ~)
    import os as _os
    _prev_cwd = _os.getcwd()
    try:
        _os.chdir(session["cwd"])
    except Exception:
        pass

    session["status"] = "running"
    try:
        # Dispatch to modifier handler if one is active, otherwise run normally
        mod_result = apply_modifiers(mods, clean_message or message, ag)
        if mod_result is not None:
            result = mod_result
            # Save non-empty modifier results to session agent history for next-message context
            # (@auto/@reason sync via worker._history; this handles @multi and similar)
            if result and result.strip():
                ag._history.append({"user": clean_message or message, "assistant": result})
        elif any(m in PASSTHROUGH_MODIFIERS for m in mods):
            # @insecure and similar passthrough modifiers: run agent normally
            result = ag.run(clean_message or message)
        else:
            result = ag.run(message)

        session["status"] = "idle"
        if result and result.strip():
            put(_evt("response", text=result))
        # Modifier workers emit via display.print_response; if still nothing was shown, ack completion
        elif mod_result is not None and not result:
            put(_evt("info", text="✓ Done."))
    except InterruptedError:
        session["status"] = "idle"
        put(_evt("info", text="⏹ Agent stopped by user"))
    except Exception as exc:
        session["status"] = "error"
        put(_evt("error", message=str(exc)))
    finally:
        # Persist any cwd change the agent made (e.g. via bash cd) back to session
        session["cwd"] = _os.getcwd()
        _os.chdir(_prev_cwd)
        _restore_display(_saved_display)
        session["stop_requested"] = False

        # Process next queued message if any (and not errored)
        queued = session.get("queued_messages", [])
        if queued and session["status"] != "error":
            next_msg = session["queued_messages"].pop(0)
            put(_evt("info", text="▸ Running queued message…"))
            threading.Thread(
                target=_run_agent, args=(session_id, next_msg, loop), daemon=True
            ).start()
        else:
            put(_evt("done"))


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    name: Optional[str] = None


class SendMessageRequest(BaseModel):
    message: str


@app.post("/sessions", status_code=201)
async def create_session(req: CreateSessionRequest):
    from agent import Agent
    import os
    session_id = uuid.uuid4().hex[:8]
    ag = Agent()
    ag.MAX_TOOL_ITERATIONS = 30
    ag.MAX_CONTINUATION_ITERATIONS = 15
    # Web sessions start in the user's home directory
    session_cwd = str(Path.home())
    sessions[session_id] = {
        "id": session_id,
        "name": req.name or f"Agent {len(sessions) + 1}",
        "status": "idle",
        "created_at": _now(),
        "cwd": session_cwd,
        "agent": ag,
        "pending_confirm": None,
        "pending_ask_user": None,
        "stop_requested": False,
        "queued_messages": [],
        "history": [],
    }
    event_queues[session_id] = asyncio.Queue()
    return _session_summary(session_id)


@app.get("/sessions")
async def list_sessions():
    return [_session_summary(sid) for sid in sessions]


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    return _session_summary(session_id)


@app.post("/sessions/{session_id}/message")
async def send_message(session_id: str, req: SendMessageRequest):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    session = sessions[session_id]

    if session["status"] == "running":
        # Queue the message instead of rejecting
        session.setdefault("queued_messages", []).append(req.message)
        ev = _evt("user_message_queued", text=req.message)
        session["history"].append(ev)
        await event_queues[session_id].put(ev)
        return {"status": "queued"}

    session["stop_requested"] = False
    loop = asyncio.get_event_loop()
    ev = _evt("user_message", text=req.message)
    session["history"].append(ev)
    await event_queues[session_id].put(ev)

    thread = threading.Thread(
        target=_run_agent,
        args=(session_id, req.message, loop),
        daemon=True,
    )
    thread.start()
    return {"status": "started"}


@app.post("/sessions/{session_id}/stop", status_code=200)
async def stop_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    session = sessions[session_id]
    session["stop_requested"] = True
    session["queued_messages"] = []  # cancel queued too
    return {"status": "stopping"}


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    sessions.pop(session_id, None)
    event_queues.pop(session_id, None)


def _session_summary(session_id: str) -> dict:
    s = sessions[session_id]
    return {
        "id": s["id"],
        "name": s["name"],
        "status": s["status"],
        "created_at": s["created_at"],
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/sessions/{session_id}/ws")
async def ws_session(websocket: WebSocket, session_id: str):
    if session_id not in sessions:
        await websocket.close(code=4004)
        return

    await websocket.accept()
    queue = event_queues[session_id]
    session = sessions[session_id]

    # Send initial handshake + history replay
    await websocket.send_json(_evt("connected", session=_session_summary(session_id)))
    if session["history"]:
        await websocket.send_json({"type": "history", "events": session["history"]})

    async def sender():
        """Forward queued agent events to the browser."""
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=25)
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})

    async def receiver():
        """Handle messages from the browser (confirmations, ask_user answers)."""
        while True:
            try:
                raw = await websocket.receive_text()
                data = json.loads(raw)
            except (WebSocketDisconnect, RuntimeError):
                return

            msg_type = data.get("type")

            if msg_type == "confirm_response":
                allow = data.get("allow", True)
                pending = session.get("pending_confirm")
                if pending:
                    pending.resolve(allow)
                # Mark resolved in history
                for evt in reversed(session["history"]):
                    if evt.get("type") == "confirm_request" and not evt.get("resolved"):
                        evt["resolved"] = True
                        evt["allowed"] = allow
                        break

            elif msg_type == "ask_user_response":
                answer = data.get("answer", "")
                pending = session.get("pending_ask_user")
                if pending:
                    pending.resolve(answer)
                # Mark answered in history
                for evt in reversed(session["history"]):
                    if evt.get("type") == "ask_user" and not evt.get("answered"):
                        evt["answered"] = True
                        evt["answer"] = answer
                        break

    try:
        await asyncio.gather(sender(), receiver())
    except (WebSocketDisconnect, Exception):
        pass


# ---------------------------------------------------------------------------
# Static files / SPA
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/", response_class=FileResponse)
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
