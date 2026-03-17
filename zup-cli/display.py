"""Terminal output helpers — mimics Claude Code's visual style."""

import json
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner as RichSpinner
from rich.text import Text
from rich import box

console = Console()

# ---------------------------------------------------------------------------
# Spinner (shown while waiting for API / tool execution)
# ---------------------------------------------------------------------------

_live: Live | None = None


def _stop_live() -> None:
    global _live
    if _live is not None:
        _live.stop()
        _live = None


def spinner_start(text: str = "Thinking…") -> None:
    global _live
    _stop_live()
    _live = Live(
        RichSpinner("dots", text=f" [dim]{text}[/dim]"),
        console=console,
        refresh_per_second=15,
        transient=True,
    )
    _live.start()


def spinner_stop() -> None:
    _stop_live()


def print_welcome(model_name: str | None = None):
    if model_name:
        model_line = f"\n[dim]Model:[/dim] [bold white]{model_name}[/bold white]"
    else:
        model_line = "\n[dim]Model: none selected \u2014 use /model to pick one[/dim]"
    console.print(
        Panel.fit(
            "[bold cyan]Zup CLI[/bold cyan]  [dim]AI coding assistant \u00b7 StackSpot AI[/dim]\n"
            "[dim]Type a message, or /help for commands. Ctrl+C to interrupt.[/dim]"
            + model_line,
            box=box.ROUNDED,
            border_style="cyan",
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
