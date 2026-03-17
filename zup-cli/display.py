"""Terminal output helpers — mimics Claude Code's visual style."""

import json
import time
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich import box

console = Console()

# ---------------------------------------------------------------------------
# Spinner (shown while waiting for API / tool execution)
# ---------------------------------------------------------------------------

_live: Live | None = None


# ---------------------------------------------------------------------------
# Streaming live view
# ---------------------------------------------------------------------------

class _StreamingView:
    """Live renderable that shows a rolling preview of streamed tokens + token count."""

    _FRAMES = ["✻", "✼", "✽", "✾", "✽", "✼"]
    _PREVIEW_LINES = 5

    def __init__(self) -> None:
        self.text = ""
        self.in_tokens = 0
        self.out_tokens = 0
        self._start = time.monotonic()
        self._tick = 0

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        frame = self._FRAMES[self._tick % len(self._FRAMES)]
        self._tick += 1
        elapsed = _fmt_elapsed(time.monotonic() - self._start)

        header = Text()
        header.append(f"{frame} ", style="bold yellow")
        header.append("Generating…", style="bold white")
        header.append(f" ({elapsed})", style="dim")
        if self.out_tokens:
            header.append(
                f"  [in:{self.in_tokens} out:{self.out_tokens}]",
                style="dim cyan",
            )
        yield header

        if self.text:
            lines = self.text.splitlines()
            shown = lines[-self._PREVIEW_LINES:]
            for line in shown:
                t = Text("  " + line, style="color(8)", overflow="fold")
                yield t


_stream_view: _StreamingView | None = None


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    return f"{m}m {s}s"


class _StatusSpinner:
    """Renders:  ✻ label (elapsed · status)"""

    _FRAMES = ["✻", "✼", "✽", "✾", "✽", "✼"]

    def __init__(self, label: str, status: str) -> None:
        self._label = label
        self._status = status
        self._start = time.monotonic()
        self._tick = 0

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        frame = self._FRAMES[self._tick % len(self._FRAMES)]
        self._tick += 1
        elapsed = _fmt_elapsed(time.monotonic() - self._start)
        t = Text()
        t.append(f"{frame} ", style="bold yellow")
        t.append(self._label, style="bold white")
        t.append(f" ({elapsed}", style="dim")
        if self._status:
            t.append(f" · {self._status}", style="dim")
        t.append(")", style="dim")
        yield t


def _stop_live() -> None:
    global _live
    if _live is not None:
        _live.stop()
        _live = None


def spinner_start(label: str = "Thinking…", status: str = "thinking") -> None:
    global _live
    _stop_live()
    _live = Live(
        _StatusSpinner(label, status),
        console=console,
        refresh_per_second=8,
        transient=True,
    )
    _live.start()


def spinner_stop() -> None:
    _stop_live()


def stream_start() -> None:
    """Begin a live streaming view (replaces spinner during LLM generation)."""
    global _live, _stream_view
    _stop_live()
    _stream_view = _StreamingView()
    _live = Live(
        _stream_view,
        console=console,
        refresh_per_second=12,
        transient=True,
    )
    _live.start()


def stream_chunk(text: str) -> None:
    """Append a streamed token chunk to the live view."""
    if _stream_view is not None:
        _stream_view.text += text


def stream_tokens(input_tokens: int, output_tokens: int) -> None:
    """Update the token counters in the live view."""
    if _stream_view is not None:
        _stream_view.in_tokens = input_tokens
        _stream_view.out_tokens = output_tokens


def stream_stop() -> None:
    """Stop the streaming live view."""
    global _stream_view
    _stop_live()
    _stream_view = None


def print_welcome(model_name: str | None = None):
    import getpass
    import os
    from pathlib import Path
    from rich.table import Table

    # Load icon
    icon_path = Path(__file__).parent / "assets" / "icon.txt"
    try:
        icon_lines = icon_path.read_text(encoding="utf-8").rstrip().splitlines()
    except Exception:
        icon_lines = []

    # Username + config
    username = getpass.getuser()
    try:
        from config import get_config
        cfg = get_config()
        realm = cfg.get("realm", "")
    except Exception:
        realm = ""

    try:
        cwd = "~" + os.sep + str(Path(os.getcwd()).relative_to(Path.home()))
    except ValueError:
        cwd = os.getcwd()

    # ── Left column ──────────────────────────────────────────────────────────
    left = Text(justify="center")
    left.append("\n")
    if icon_lines:
        for line in icon_lines:
            left.append(line + "\n", style="bold cyan")
    left.append("\n")

    # ── Right column ─────────────────────────────────────────────────────────
    right = Text()
    right.append("\n")
    right.append("Tips for getting started\n", style="bold cyan")
    right.append("  Run ", style="")
    right.append("/help", style="bold white")
    right.append(" to see all available commands\n", style="")
    right.append("  Type any message and press ", style="dim")
    right.append("Enter", style="bold dim")
    right.append(" to chat with the AI\n", style="dim")
    right.append("  Use ", style="dim")
    right.append("/model", style="bold dim")
    right.append(" or ", style="dim")
    right.append("/agent", style="bold dim")
    right.append(" to switch AI settings\n", style="dim")
    right.append("  Use ", style="dim")
    right.append("/ks", style="bold dim")
    right.append(" to manage knowledge sources\n", style="dim")
    right.append("\u2500" * 44 + "\n", style="dim")
    right.append(f"\nWelcome back {username}!\n", style="bold white")
    meta_parts = []
    if model_name:
        meta_parts.append(model_name)
    if realm:
        meta_parts.append(realm)
    if meta_parts:
        right.append(" · ".join(meta_parts) + "\n", style="dim")
    right.append(cwd + "\n", style="dim")

    # ── Two-column grid ───────────────────────────────────────────────────────
    grid = Table.grid(padding=(0, 2))
    grid.add_column(min_width=28, ratio=1)
    grid.add_column(ratio=2)
    grid.add_row(left, right)

    console.print(
        Panel(
            grid,
            title="[bold]Zup CLI[/bold]",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(0, 1),
        )
    )
    console.print()


def print_thinking(text: str) -> None:
    """Render chain-of-thought in very dim style — like Claude's thinking blocks."""
    if not text.strip():
        return
    console.print()
    console.print("[dim]◆ Thinking[/dim]", highlight=False)
    for line in text.strip().splitlines():
        console.print(f"[color(8)]  {line}[/color(8)]")
    console.print(Rule(style="color(8)"))


def print_tool_use(name: str, parameters: dict) -> None:
    params_preview = {}
    for k, v in parameters.items():
        if isinstance(v, str) and len(v) > 60:
            params_preview[k] = v[:57] + "..."
        else:
            params_preview[k] = v
    params_str = ", ".join(f"{k}: {json.dumps(v)}" for k, v in params_preview.items())
    console.print(f"  [bold cyan]●[/bold cyan] [cyan]{name}[/cyan]({params_str})")


def print_tool_result(name: str, result: str) -> None:
    lines = result.splitlines()
    preview = lines[0][:120] if lines else ""
    extra = f" [dim](+{len(lines)-1} lines)[/dim]" if len(lines) > 1 else ""
    console.print(f"  [bold green]✔[/bold green] [dim]{preview}{extra}[/dim]")


def print_response(text: str):
    """Render the assistant's final response as markdown."""
    if not text.strip():
        return
    console.print()
    try:
        console.print(Markdown(text))
    except Exception:
        console.print(text)
    console.print()


def print_error(message: str):
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_info(message: str):
    console.print(f"[dim]{message}[/dim]")


def print_separator() -> None:
    console.print(Rule(style="dim"))
