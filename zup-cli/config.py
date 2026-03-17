import json
from pathlib import Path
from typing import Optional

CONFIG_DIR = Path.home() / ".zup-cli"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_AGENT_ID = "01J4WFMAJTP453TRRQKFAJ80PN"
DEFAULT_REALM = "zup"


def get_config() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE) as f:
        return json.load(f)


def save_config(config: dict):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_required(key: str) -> str:
    val = get_config().get(key)
    if not val:
        raise RuntimeError(
            f"Missing config '{key}'. Run: python main.py --config"
        )
    return val


def configure():
    from rich.console import Console
    from rich.prompt import Prompt

    console = Console()
    console.print("\n[bold cyan]Zup CLI Setup[/bold cyan]")
    console.print("Configure your StackSpot AI credentials.\n")

    existing = get_config()

    client_id = Prompt.ask(
        "Client ID", default=existing.get("client_id", "")
    )
    client_secret = Prompt.ask(
        "Client Secret", default=existing.get("client_secret", ""), password=True
    )
    realm = Prompt.ask("Realm", default=existing.get("realm", DEFAULT_REALM))
    agent_id = Prompt.ask(
        "Agent ID", default=existing.get("agent_id", DEFAULT_AGENT_ID)
    )

    config = {
        "client_id": client_id,
        "client_secret": client_secret,
        "realm": realm,
        "agent_id": agent_id,
    }
    save_config(config)
    console.print("\n[green]✓ Configuration saved to ~/.zup-cli/config.json[/green]\n")
