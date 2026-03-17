"""Interactive REPL and slash-command handling."""

import os
import sys
from pathlib import Path
from typing import Optional

from prompt_toolkit.completion import Completer, Completion

import display
from agent import Agent


# ---------------------------------------------------------------------------
# Slash-command autocomplete
# ---------------------------------------------------------------------------

# All completable slash tokens, in display order.
# Each entry: (full_text_to_complete, display_label, description)
_SLASH_COMPLETIONS: list[tuple[str, str, str]] = [
    ("/help",             "/help",                    "Show available commands"),
    ("/reset",            "/reset",                   "Start a new conversation (new ID)"),
    ("/debug",            "/debug",                   "Show debug log path"),
    ("/cwd",              "/cwd [path]",               "Show or change working directory"),
    ("/model",            "/model",                   "Switch model interactively"),
    ("/models",           "/models",                  "List all available AI models"),
    ("/ks list",          "/ks list [page] [size]",    "List knowledge sources"),
    ("/ks objects",       "/ks objects <slug> [page]", "List objects in a knowledge source"),
    ("/ks details",       "/ks details <slug>",        "Get knowledge source details"),
    ("/ks create",        "/ks create <name> <slug>",  "Create a new knowledge source"),
    ("/ks upload",        "/ks upload <file> <slug>",  "Upload a file to a knowledge source"),
    ("/exit",             "/exit",                    "Quit"),
]


class SlashCompleter(Completer):
    """Complete slash commands as the user types."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Only activate when the line starts with /
        if not text.startswith("/"):
            return

        for token, label, desc in _SLASH_COMPLETIONS:
            if token.startswith(text):
                # Yield only the remaining suffix
                yield Completion(
                    token[len(text):],
                    start_position=0,
                    display=label,
                    display_meta=desc,
                )

HISTORY_FILE = Path.home() / ".zup-cli" / "history"


def _pick_model(current_id: Optional[str]) -> Optional[tuple[str, str]]:
    """
    Show an interactive arrow-key selection list of available models.
    Returns the selected model ID, or None if cancelled.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style as PTStyle
    from rich.console import Console

    con = Console()

    try:
        from api_client import list_models
        raw = list_models()
    except Exception as e:
        display.print_error(f"Could not fetch models: {e}")
        return None

    # Normalise to list
    if isinstance(raw, dict):
        items = (
            raw.get("items") or raw.get("data") or
            raw.get("models") or raw.get("content") or []
        )
    else:
        items = raw if isinstance(raw, list) else []

    if not items:
        display.print_info("No models available.")
        return None

    # Build (id, display_name) pairs
    entries: list[tuple[str, str]] = []
    for m in items:
        mid = (
            m.get("id") or m.get("model_id") or
            m.get("modelId") or m.get("slug") or ""
        )
        mname = (
            m.get("display_name") or m.get("displayName") or
            m.get("name") or m.get("title") or mid
        )
        if mid:
            entries.append((mid, mname))

    if not entries:
        display.print_info("No selectable models found.")
        return None

    state = {"index": 0, "result": None, "done": False}
    # Start cursor on current model if any
    if current_id:
        for i, (mid, _) in enumerate(entries):
            if mid == current_id:
                state["index"] = i
                break

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(event):
        state["index"] = (state["index"] - 1) % len(entries)

    @kb.add("down")
    @kb.add("j")
    def _down(event):
        state["index"] = (state["index"] + 1) % len(entries)

    @kb.add("enter")
    def _select(event):
        state["result"] = entries[state["index"]]  # (id, display_name)
        state["done"] = True
        event.app.exit()

    @kb.add("c-c")
    @kb.add("escape")
    @kb.add("q")
    def _cancel(event):
        state["done"] = True
        event.app.exit()

    def _get_text():
        lines = [("class:title", " Select model  ↑/↓ or j/k · Enter to confirm · Esc to cancel\n\n")]
        for i, (mid, mname) in enumerate(entries):
            selected = i == state["index"]
            marker = "  ● " if selected else "  ○ "
            cursor = "class:selected" if selected else "class:item"
            current = "class:current" if mid == current_id else ""
            tag = " [current]" if mid == current_id else ""
            lines.append((cursor, marker))
            lines.append((cursor, mname))
            if current:
                lines.append((current, tag))
            lines.append(("", f"   {mid}\n" if not selected else f"   {mid}\n"))
        return lines

    style = PTStyle.from_dict({
        "title":    "bold cyan",
        "selected": "bold white reverse",
        "item":     "",
        "current":  "dim green",
    })

    layout = Layout(
        HSplit([
            Window(content=FormattedTextControl(_get_text, focusable=True)),
        ])
    )

    app: Application = Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=False,
    )
    app.run()

    return state["result"]


def _confirm_tool(name: str, parameters: dict) -> bool:
    """
    Show an interactive Accept / Decline prompt before executing a mutating tool.
    Returns True to allow, False to deny.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style as PTStyle

    # Build a human-readable description of what the tool will do
    preview_lines: list[str] = []
    if name == "write_file":
        path = parameters.get("path", "?")
        content = parameters.get("content", "")
        lines = content.splitlines()
        shown = lines[:20]
        preview_lines = [f"  Write file: {path}", ""]
        preview_lines += [f"    {ln}" for ln in shown]
        if len(lines) > 20:
            preview_lines.append(f"    ... ({len(lines) - 20} more lines)")
    elif name == "edit_file":
        path = parameters.get("path", "?")
        old_str = parameters.get("old_str", "")
        new_str = parameters.get("new_str", "")
        preview_lines = [
            f"  Edit file: {path}", "",
            "  Replace:",
        ] + [f"  - {ln}" for ln in old_str.splitlines()] + [
            "",
            "  With:",
        ] + [f"  + {ln}" for ln in new_str.splitlines()]
    elif name == "bash":
        cmd = parameters.get("command", "?")
        preview_lines = [f"  Run command:", f"    {cmd}"]
    else:
        preview_lines = [f"  {name}: {parameters}"]

    OPTIONS = ["Accept", "Decline"]
    state = {"index": 0, "result": None}

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(event):
        state["index"] = (state["index"] - 1) % len(OPTIONS)

    @kb.add("down")
    @kb.add("j")
    def _down(event):
        state["index"] = (state["index"] + 1) % len(OPTIONS)

    @kb.add("enter")
    def _select(event):
        state["result"] = OPTIONS[state["index"]]
        event.app.exit()

    @kb.add("a")
    @kb.add("y")
    def _accept(event):
        state["result"] = "Accept"
        event.app.exit()

    @kb.add("d")
    @kb.add("n")
    @kb.add("escape")
    @kb.add("c-c")
    def _decline(event):
        state["result"] = "Decline"
        event.app.exit()

    def _get_text():
        lines = [("class:title", f"\n  Confirm action: {name}\n\n")]
        for ln in preview_lines:
            lines.append(("class:preview", ln + "\n"))
        lines.append(("", "\n  ↑/↓ or a/d · Enter to confirm\n\n"))
        for i, opt in enumerate(OPTIONS):
            selected = i == state["index"]
            marker = "  ● " if selected else "  ○ "
            style = "class:selected" if selected else "class:option"
            lines.append((style, f"{marker}{opt}\n"))
        return lines

    style = PTStyle.from_dict({
        "title":    "bold yellow",
        "preview":  "cyan",
        "selected": "bold white reverse",
        "option":   "",
    })

    layout = Layout(
        HSplit([Window(content=FormattedTextControl(_get_text, focusable=True))])
    )

    app: Application = Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=False,
    )
    app.run()

    return state["result"] == "Accept"


def _print_models(data):
    """Render the models list: bold display name + dim ID."""
    from rich.console import Console
    from rich.text import Text

    con = Console()

    # The API may return a list directly or a dict with an 'items'/'data' key.
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = (
            data.get("items")
            or data.get("data")
            or data.get("models")
            or data.get("content")
            or []
        )
    else:
        items = []

    if not items:
        con.print("[dim]No models found.[/dim]")
        return

    con.print(f"\n[bold cyan]Available models[/bold cyan] ({len(items)})\n")
    for model in items:
        # Field names vary across APIs; try common variants.
        display_name = (
            model.get("display_name")
            or model.get("displayName")
            or model.get("name")
            or model.get("title")
            or "Unknown"
        )
        model_id = (
            model.get("id")
            or model.get("model_id")
            or model.get("modelId")
            or model.get("slug")
            or ""
        )

        line = Text()
        line.append(f"  {display_name}", style="bold white")
        if model_id:
            line.append(f"  {model_id}", style="dim")
        con.print(line)

    con.print()


HELP_TEXT = """
[bold cyan]Commands[/bold cyan]
  /help                         Show this help
  /reset                        Start a new conversation (generates a new ID)
  /debug                        Show debug log path (start with --debug to enable)
  /cwd [path]                   Show or change working directory
  /models                       List available AI models
  /ks list [page] [size]        List knowledge sources
  /ks objects <slug> [page]     List objects inside a knowledge source
  /ks details <slug>            Get knowledge source details
  /ks create <name> <slug> [description...]
                                Create a new knowledge source
  /ks upload <file> <slug>      Upload a local file to a knowledge source
  /exit                         Quit

[bold cyan]Usage[/bold cyan]
  Type any message and press Enter. The AI reads/writes files,
  runs shell commands, and manages knowledge sources as needed.
"""


def _handle_slash(cmd: str, agent: Agent) -> bool:
    """Handle a /command. Returns True if handled."""
    parts = cmd.strip().split()
    if not parts:
        return False

    head = parts[0].lower()

    if head == "/help":
        from rich.console import Console
        Console().print(HELP_TEXT)
        return True

    if head in ("/reset", "/clear"):
        agent.reset()
        display.print_info(f"Conversation reset. New ID: {agent.conversation_id}")
        return True

    if head == "/debug":
        import logger
        if logger.is_enabled():
            display.print_info(f"Debug logging active → {logger._log_path.resolve()}")
        else:
            display.print_info("Debug mode is OFF. Restart with --debug to enable.")
        return True

    if head == "/cwd":
        if len(parts) > 1:
            try:
                os.chdir(parts[1])
                display.print_info(f"Working directory: {os.getcwd()}")
            except Exception as e:
                display.print_error(str(e))
        else:
            display.print_info(f"Working directory: {os.getcwd()}")
        return True

    if head in ("/exit", "/quit", "/q"):
        display.print_info("Goodbye!")
        sys.exit(0)

    if head == "/models":
        try:
            from api_client import list_models
            data = list_models()
            _print_models(data)
        except Exception as e:
            display.print_error(str(e))
        return True

    if head == "/model":
        chosen = _pick_model(agent.selected_model)
        if chosen:
            # chosen is (id, display_name)
            model_id, model_name = chosen
            agent.set_model(model_id, model_name)
            display.print_info(f"Model set to: {model_name}")
        else:
            display.print_info("No model selected.")
        return True

    if head == "/ks":
        if len(parts) < 2:
            display.print_info("Usage: /ks [list|objects|details|create|upload]")
            return True

        sub = parts[1].lower()

        if sub == "list":
            page = int(parts[2]) if len(parts) > 2 else 1
            size = int(parts[3]) if len(parts) > 3 else 10
            from tools import list_knowledge_sources_tool
            display.console.print(list_knowledge_sources_tool(page=page, size=size))
            return True

        if sub == "objects":
            if len(parts) < 3:
                display.print_info("Usage: /ks objects <slug> [page]")
                return True
            slug = parts[2]
            page = int(parts[3]) if len(parts) > 3 else 1
            from tools import get_ks_objects_tool
            display.console.print(get_ks_objects_tool(slug=slug, page=page))
            return True

        if sub == "details":
            if len(parts) < 3:
                display.print_info("Usage: /ks details <slug>")
                return True
            from tools import get_ks_details_tool
            display.console.print(get_ks_details_tool(slug=parts[2]))
            return True

        if sub == "create":
            if len(parts) < 4:
                display.print_info("Usage: /ks create <name> <slug> [description...]")
                return True
            name, slug = parts[2], parts[3]
            desc = " ".join(parts[4:]) if len(parts) > 4 else ""
            from tools import create_ks_tool
            display.console.print(create_ks_tool(name=name, slug=slug, description=desc))
            return True

        if sub == "upload":
            if len(parts) < 4:
                display.print_info("Usage: /ks upload <file> <slug>")
                return True
            file_path, ks_slug = parts[2], parts[3]
            display.print_info(f"Uploading {file_path} → {ks_slug} ...")
            from tools import upload_to_ks_tool
            display.console.print(upload_to_ks_tool(file_path=file_path, ks_slug=ks_slug))
            return True

        display.print_info("Unknown /ks subcommand. Try /help.")
        return True

    return False  # unrecognised command


def _process(message: str, agent: Agent):
    """Run the agent for one user message and display the result."""
    import logger
    logger.log_user_input(message)

    # --- spinner-aware callback wrappers -----------------------------------
    _orig_thinking    = agent.on_thinking
    _orig_tool_use    = agent.on_tool_use
    _orig_tool_result = agent.on_tool_result
    _orig_confirm     = agent.on_confirm_tool

    def _on_thinking(text: str):
        display.spinner_stop()
        _orig_thinking(text)
        display.spinner_start("Thinking…", status="thinking")

    def _on_tool_use(name: str, params: dict):
        display.spinner_stop()
        _orig_tool_use(name, params)
        display.spinner_start(f"Running {name}…", status=name)

    def _on_tool_result(name: str, result: str):
        display.spinner_stop()
        _orig_tool_result(name, result)
        display.spinner_start("Thinking…", status="thinking")

    def _on_confirm(name: str, params: dict) -> bool:
        display.spinner_stop()
        return _orig_confirm(name, params)

    agent.on_thinking    = _on_thinking
    agent.on_tool_use    = _on_tool_use
    agent.on_tool_result = _on_tool_result
    agent.on_confirm_tool = _on_confirm
    # -----------------------------------------------------------------------

    display.spinner_start("Thinking…", status="thinking")
    try:
        response = agent.run(message)
        logger.log_agent_response(response)
        display.spinner_stop()
        display.print_separator()
        display.print_response(response)
    except KeyboardInterrupt:
        display.spinner_stop()
        display.console.print("\n[yellow]Interrupted.[/yellow]")
    except Exception as e:
        display.spinner_stop()
        logger.log_error("_process", e)
        display.print_error(str(e))
        # Invalidate token if it looks like an auth error
        if "401" in str(e) or "403" in str(e) or "auth" in str(e).lower():
            from auth import invalidate_token
            invalidate_token()
            display.print_info("Token cleared — will re-authenticate on next request.")
    finally:
        # Restore original callbacks so /reset works correctly
        agent.on_thinking     = _orig_thinking
        agent.on_tool_use     = _orig_tool_use
        agent.on_tool_result  = _orig_tool_result
        agent.on_confirm_tool = _orig_confirm


def start_repl(initial_prompt: Optional[str] = None):
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style

    # Verify credentials up-front
    try:
        from auth import get_token
        get_token()
    except Exception as e:
        display.print_error(f"Authentication failed: {e}")
        display.print_info("Run:  python main.py --config")
        sys.exit(1)

    agent = Agent(
        on_tool_use=display.print_tool_use,
        on_tool_result=display.print_tool_result,
        on_thinking=display.print_thinking,
        on_confirm_tool=_confirm_tool,
    )

    display.print_welcome(model_name=agent.selected_model_name)
    display.print_info(f"Authenticated ✓  |  conversation: {agent.conversation_id}\n")

    import logger
    if logger.is_enabled():
        display.print_info(f"Debug mode ON — logging to {logger._log_path.resolve()}\n")

    # Non-interactive mode (pipe / single prompt)
    if initial_prompt:
        _process(initial_prompt, agent)
        if not sys.stdin.isatty():
            return

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        style=Style.from_dict({"prompt": "bold cyan"}),
        completer=SlashCompleter(),
        complete_while_typing=True,
    )

    while True:
        try:
            cwd_short = os.path.basename(os.getcwd()) or os.getcwd()
            model_tag = (
                f" [{agent.selected_model_name}]" if agent.selected_model_name else ""
            )
            user_input = session.prompt(
                [("class:prompt", f"{cwd_short}{model_tag}> ")]
            ).strip()
        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            display.print_info("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if not _handle_slash(user_input, agent):
                display.print_info(f"Unknown command '{user_input}'. Try /help.")
            continue

        _process(user_input, agent)
