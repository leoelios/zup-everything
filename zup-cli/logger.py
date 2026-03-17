"""
Debug logger — writes a structured trace of every CLI interaction to logs.txt.

Usage:
  python main.py --debug          # logs to ./logs.txt
  python main.py --debug my prompt
"""

import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

_enabled = False
_log_path: Path = Path("logs.txt")
_log_file = None


def setup(path: str = "logs.txt"):
    """Enable debug logging. Called once at startup when --debug is passed."""
    global _enabled, _log_path, _log_file
    _enabled = True
    _log_path = Path(path)
    _log_file = _log_path.open("a", encoding="utf-8")
    _write_separator()
    _raw(f"SESSION START  {datetime.now().isoformat(timespec='seconds')}  pid={os.getpid()}")
    _raw(f"cwd: {os.getcwd()}")
    _raw(f"python: {sys.version.split()[0]}")
    _write_separator()


def is_enabled() -> bool:
    return _enabled


def _write_separator():
    _raw("=" * 80)


def _raw(line: str):
    if _log_file:
        _log_file.write(line + "\n")
        _log_file.flush()


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _block(category: str, body: str):
    """Write a labelled block."""
    if not _enabled:
        return
    _raw(f"\n[{_ts()}] [{category}]")
    for line in body.splitlines():
        _raw(f"  {line}")


def log_user_input(text: str):
    _block("USER INPUT", text)


def log_api_request(prompt: str, conversation_id: str | None, model: str | None, streaming: bool):
    if not _enabled:
        return
    _raw(f"\n[{_ts()}] [API REQUEST]")
    _raw(f"  conversation_id : {conversation_id!r}")
    _raw(f"  model           : {model!r}")
    _raw(f"  streaming       : {streaming}")
    _raw(f"  --- prompt ({len(prompt)} chars) ---")
    for line in prompt.splitlines():
        _raw(f"  {line}")
    _raw(f"  --- end prompt ---")


def log_api_response(response: dict):
    if not _enabled:
        return
    _raw(f"\n[{_ts()}] [API RESPONSE]")
    _raw(f"  conversation_id : {response.get('conversation_id')!r}")
    message = response.get("message", "")
    _raw(f"  --- message ({len(message)} chars) ---")
    for line in message.splitlines():
        _raw(f"  {line}")
    _raw(f"  --- end message ---")
    # Log any extra keys (status, model, etc.) except the big fields
    extras = {k: v for k, v in response.items() if k not in ("message", "conversation_id")}
    if extras:
        _raw(f"  extras: {extras}")


def log_thinking(text: str):
    _block("THINKING", text)


def log_tool_call(name: str, parameters: dict):
    import json
    if not _enabled:
        return
    _raw(f"\n[{_ts()}] [TOOL CALL] {name}")
    try:
        _raw(f"  params: {json.dumps(parameters, ensure_ascii=False)}")
    except Exception:
        _raw(f"  params: {parameters!r}")


def log_tool_result(name: str, result: str):
    _block(f"TOOL RESULT  {name}", result)


def log_tool_confirm(name: str, accepted: bool):
    _block("TOOL CONFIRM", f"{name} -> {'ACCEPTED' if accepted else 'DECLINED'}")


def log_agent_response(text: str):
    _block("AGENT RESPONSE", text)


def log_error(context: str, error: Any):
    _block(f"ERROR  {context}", str(error))


def close():
    global _log_file
    if _log_file:
        _write_separator()
        _raw(f"SESSION END  {datetime.now().isoformat(timespec='seconds')}")
        _write_separator()
        _log_file.close()
        _log_file = None
