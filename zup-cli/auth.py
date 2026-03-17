import json
import time
import httpx
from pathlib import Path
from config import get_config, CONFIG_DIR

TOKEN_FILE = CONFIG_DIR / "token.json"
IDM_BASE = "https://idm.stackspot.com"


def _load_cached() -> dict | None:
    if not TOKEN_FILE.exists():
        return None
    with open(TOKEN_FILE) as f:
        data = json.load(f)
    # Keep 60s buffer before expiry
    if data.get("expires_at", 0) > time.time() + 60:
        return data
    return None


def _save_token(data: dict):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data["expires_at"] = time.time() + data.get("expires_in", 1199)
    with open(TOKEN_FILE, "w") as f:
        json.dump(data, f)


def get_token() -> str:
    cached = _load_cached()
    if cached:
        return cached["access_token"]

    cfg = get_config()
    if not cfg.get("client_id") or not cfg.get("client_secret"):
        raise RuntimeError(
            "Not configured. Run: python main.py --config"
        )

    realm = cfg.get("realm", "zup")
    url = f"{IDM_BASE}/{realm}/oidc/oauth/token"

    resp = httpx.post(
        url,
        data={
            "grant_type": "client_credentials",
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
        },
        timeout=30.0,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Auth failed ({resp.status_code}): {resp.text}")

    data = resp.json()
    _save_token(data)
    return data["access_token"]


def invalidate_token():
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()
