import json
import time
import httpx
from typing import Generator, Optional
from auth import get_token
from config import get_config, DEFAULT_AGENT_ID

CHAT_BASE = "https://genai-inference-app.stackspot.com"
DATA_BASE = "https://data-integration-api.stackspot.com"
KS_BASE = DATA_BASE


def _headers(accept: str = "application/json") -> dict:
    return {
        "Authorization": f"Bearer {get_token()}",
        "Content-Type": "application/json",
        "Accept": accept,
        "x-request-origin": "chat",
    }


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def chat_nonstream(
    prompt: str,
    conversation_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    selected_model: Optional[str] = None,
) -> dict:
    cfg = get_config()
    aid = agent_id or cfg.get("agent_id", DEFAULT_AGENT_ID)
    url = f"{CHAT_BASE}/v1/agent/{aid}/chat"

    payload: dict = {
        "user_prompt": prompt,
        "streaming": False,
        "show_chat_processing_state": False,
        "return_ks_in_response": False,
        "deep_search_ks": False,
        "stackspot_knowledge": False,
        "use_conversation": True,
        "conversation_id": conversation_id,
        "upload_ids": [],
        "selected_model": selected_model,
    }

    import logger
    logger.log_api_request(prompt, conversation_id, selected_model, streaming=False)

    resp = httpx.post(url, json=payload, headers=_headers(), timeout=120.0)
    resp.raise_for_status()
    result = resp.json()
    logger.log_api_response(result)
    return result


def chat_stream(
    prompt: str,
    conversation_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    selected_model: Optional[str] = None,
) -> Generator[dict, None, None]:
    cfg = get_config()
    aid = agent_id or cfg.get("agent_id", DEFAULT_AGENT_ID)
    url = f"{CHAT_BASE}/v1/agent/{aid}/chat"

    payload: dict = {
        "user_prompt": prompt,
        "streaming": True,
        "show_chat_processing_state": False,
        "return_ks_in_response": False,
        "deep_search_ks": False,
        "stackspot_knowledge": False,
        "use_conversation": True,
        "conversation_id": conversation_id,
        "upload_ids": [],
        "selected_model": selected_model,
    }

    headers = _headers(accept="text/event-stream")

    import logger as _logger
    with httpx.stream("POST", url, json=payload, headers=headers, timeout=120.0) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            line = line.strip()
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str:
                    try:
                        chunk = json.loads(data_str)
                        _logger._block("SSE CHUNK", str(chunk)) if _logger.is_enabled() else None
                        yield chunk
                    except json.JSONDecodeError:
                        pass


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

AGENT_BASE = "https://genai-agent-tools-api.stackspot.com"


def list_agents(page: int = 1, size: int = 100, visibility: str = "personal") -> dict:
    resp = httpx.get(
        f"{AGENT_BASE}/v4/agents",
        headers=_headers(),
        params={"page": page, "size": size, "visibility_list": visibility},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def list_models() -> list:
    resp = httpx.get(
        f"{CHAT_BASE}/v1/llm/models",
        headers=_headers(),
        params={"active": "true", "page_size": "999"},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Knowledge Sources
# ---------------------------------------------------------------------------

def list_knowledge_sources(page: int = 1, size: int = 10, visibility: str = "account") -> dict:
    resp = httpx.get(
        f"{KS_BASE}/v2/knowledge-sources",
        headers=_headers(),
        params={"page": page, "size": size, "visibility_list": visibility},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def get_ks_objects(slug: str, page: int = 1, size: int = 20) -> dict:
    resp = httpx.get(
        f"{KS_BASE}/v2/knowledge-sources/{slug}/objects",
        headers=_headers(),
        params={"page": page, "size": size, "content_limit": 5000},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def get_ks_details(slug: str) -> dict:
    resp = httpx.get(
        f"{DATA_BASE}/v1/knowledge-sources/{slug}",
        headers=_headers(),
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def create_knowledge_source(
    name: str, slug: str, description: str = "", visibility: str = "personal"
) -> dict:
    resp = httpx.post(
        f"{DATA_BASE}/v1/knowledge-sources",
        headers=_headers(),
        json={
            "name": name,
            "slug": slug,
            "description": description,
            "default": False,
            "visibility_level": visibility,
            "creator": "",
            "id": "",
            "type": "custom",
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def upload_file_to_ks(local_path: str, ks_slug: str) -> dict:
    """
    3-step upload flow:
      1. POST /v2/file-upload/form      → get presigned S3 form fields + upload_id
      2. POST <s3_url> (multipart form) → upload the file with presigned fields
      3. POST /v1/file-upload/{id}/split → process/embed
      Then poll /v1/file-upload/{id} until COMPLETED.
    """
    import os

    file_name = os.path.basename(local_path)

    import logger
    import json as _json

    # Step 1: request presigned form
    _form_payload = {
        "file_name": file_name,
        "target_id": ks_slug,
        "target_type": "KNOWLEDGE_SOURCE",
        "expiration": 3600,
    }
    logger._block("KS UPLOAD step1 request", f"POST {DATA_BASE}/v2/file-upload/form\n{_json.dumps(_form_payload, indent=2)}")

    form_resp = httpx.post(
        f"{DATA_BASE}/v2/file-upload/form",
        headers=_headers(),
        json=_form_payload,
        timeout=30.0,
    )
    logger._block("KS UPLOAD step1 response", f"status={form_resp.status_code}\n{form_resp.text}")
    form_resp.raise_for_status()
    form_data = form_resp.json()

    upload_url: str = form_data.get("url") or form_data.get("upload_url", "")
    upload_id: str = (
        form_data.get("upload_id")
        or form_data.get("id")
        or form_data.get("file_upload_id", "")
    )
    # presigned S3 form fields (key, x-amz-*, policy, etc.)
    fields: dict = form_data.get("form") or form_data.get("fields") or form_data.get("form_fields") or {}

    logger._block("KS UPLOAD step1 parsed", f"upload_url={upload_url!r}\nupload_id={upload_id!r}\nfields keys={list(fields.keys())}")

    if not upload_url:
        raise RuntimeError(f"No upload URL in form response: {form_data}")
    if not upload_id:
        raise RuntimeError(f"No upload ID in form response: {form_data}")

    # Step 2: upload via multipart form POST to S3
    with open(local_path, "rb") as f:
        content = f.read()

    # Build multipart: all presigned fields first, then the file
    multipart_data = {k: (None, v) for k, v in fields.items()}
    multipart_data["file"] = (file_name, content)

    logger._block("KS UPLOAD step2 request", f"POST {upload_url}\nmultipart fields (excl. file): {[k for k in multipart_data if k != 'file']}\nfile size={len(content)} bytes")

    s3_resp = httpx.post(upload_url, files=multipart_data, timeout=120.0)
    logger._block("KS UPLOAD step2 response", f"status={s3_resp.status_code}\n{s3_resp.text[:2000]}")
    s3_resp.raise_for_status()

    # Poll until done
    for i in range(30):
        time.sleep(3)
        poll = httpx.get(
            f"{DATA_BASE}/v1/file-upload/{upload_id}",
            headers=_headers(),
            timeout=30.0,
        )
        poll.raise_for_status()
        status = poll.json()
        state = status.get("status", "").upper()
        logger._block(f"KS UPLOAD poll [{i+1}/30]", f"state={state!r}\n{_json.dumps(status, indent=2, default=str)}")
        if state in ("COMPLETED", "EMBEDDED", "DONE", "INDEXED"):
            return status
        if state == "FAILED":
            raise RuntimeError(f"Upload processing failed: {status}")

    return {"status": "PROCESSING", "upload_id": upload_id}
