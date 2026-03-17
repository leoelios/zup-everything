"""Terminal output helpers — mimics Claude Code's visual style."""

import json
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()


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


def print_tool_use(name: str, parameters: dict):
    params_preview = {}
    for k, v in parameters.items():
        if isinstance(v, str) and len(v) > 60:
            params_preview[k] = v[:57] + "..."
        else:
            params_preview[k] = v
    params_str = ", ".join(f"{k}: {json.dumps(v)}" for k, v in params_preview.items())
    console.print(f"  [bold cyan]Tool[/bold cyan] [cyan]{name}[/cyan]({params_str})")


def print_tool_result(name: str, result: str):
    lines = result.splitlines()
    preview = lines[0][:120] if lines else ""
    extra = f" [dim](+{len(lines)-1} lines)[/dim]" if len(lines) > 1 else ""
    console.print(f"  [bold green]Done[/bold green]  {preview}{extra}")


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


def print_separator():
    console.print("[dim]─[/dim]" * 60)
