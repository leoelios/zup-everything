"""
Local tool implementations — file system, shell, knowledge sources.
All functions return a string result (success or error message).
"""

import fnmatch
import os
import re
import subprocess
from glob import glob
from typing import Optional


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------

def _resolve(path: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    return path


def read_file(path: str) -> str:
    fpath = _resolve(path)
    if not os.path.exists(fpath):
        return f"Error: file not found: {fpath}"
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        numbered = "".join(f"{i+1:6}\t{line}" for i, line in enumerate(lines))
        return f"File: {fpath} ({len(lines)} lines)\n\n{numbered}"
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str) -> str:
    fpath = _resolve(path)
    try:
        parent = os.path.dirname(fpath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {fpath}"
    except Exception as e:
        return f"Error writing file: {e}"


def edit_file(path: str, old_str: str, new_str: str) -> str:
    fpath = _resolve(path)
    if not os.path.exists(fpath):
        return f"Error: file not found: {fpath}"
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        count = content.count(old_str)
        if count == 0:
            return f"Error: string not found in {fpath}"
        if count > 1:
            return (
                f"Error: string found {count} times — provide more surrounding context "
                "to make it unique."
            )

        new_content = content.replace(old_str, new_str, 1)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"Edited {fpath}"
    except Exception as e:
        return f"Error editing file: {e}"


def list_files(path: str = ".", pattern: str = "**/*") -> str:
    dpath = _resolve(path)
    if not os.path.exists(dpath):
        return f"Error: directory not found: {dpath}"
    try:
        matches = glob(os.path.join(dpath, pattern), recursive=True)
        entries = []
        for m in sorted(matches):
            rel = os.path.relpath(m, dpath)
            if os.path.isdir(m):
                entries.append(rel + "/")
            else:
                entries.append(rel)
        if not entries:
            return f"No entries found in {dpath} matching '{pattern}'"
        # Cap at 300 to avoid huge outputs
        truncated = entries[:300]
        suffix = f"\n... ({len(entries) - 300} more)" if len(entries) > 300 else ""
        return f"Files in {dpath}:\n" + "\n".join(truncated) + suffix
    except Exception as e:
        return f"Error listing files: {e}"


def search_files(pattern: str, path: str = ".", file_glob: str = "*") -> str:
    dpath = _resolve(path)
    SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"

    results: list[str] = []
    try:
        for root, dirs, files in os.walk(dpath):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for fname in files:
                if not fnmatch.fnmatch(fname, file_glob):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if regex.search(line):
                                rel = os.path.relpath(fpath, dpath)
                                results.append(f"{rel}:{i}: {line.rstrip()}")
                                if len(results) >= 200:
                                    break
                except OSError:
                    pass
            if len(results) >= 200:
                break
    except Exception as e:
        return f"Error searching files: {e}"

    if not results:
        return f"No matches for '{pattern}' in {dpath}"
    return f"Matches for '{pattern}':\n" + "\n".join(results)


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def bash(command: str, timeout: int = 60) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
        )
        parts = []
        if result.stdout:
            parts.append(result.stdout.rstrip())
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr.rstrip()}")
        if result.returncode != 0:
            parts.append(f"[exit {result.returncode}]")
        return "\n".join(parts) if parts else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Knowledge Source tools (delegates to api_client)
# ---------------------------------------------------------------------------

def list_knowledge_sources_tool(page: int = 1, size: int = 10) -> str:
    try:
        from api_client import list_knowledge_sources

        data = list_knowledge_sources(page=page, size=size)
        items = data.get("items", [])
        total = data.get("total_pages", 1)

        lines = [f"Knowledge Sources (page {page}/{total}, {len(items)} shown):"]
        for ks in items:
            lines.append(
                f"\n  slug:    {ks['slug']}\n"
                f"  name:    {ks['name']}\n"
                f"  objects: {ks.get('object_count', 0)}  |  "
                f"creator: {ks.get('creator', 'N/A')}\n"
                f"  desc:    {(ks.get('description') or '')[:100]}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing knowledge sources: {e}"


def get_ks_objects_tool(slug: str, page: int = 1, size: int = 10) -> str:
    try:
        from api_client import get_ks_objects

        data = get_ks_objects(slug=slug, page=page, size=size)
        items = data.get("items", [])
        total = data.get("total_pages", 1)

        lines = [f"Objects in '{slug}' (page {page}/{total}):"]
        for obj in items:
            lines.append(
                f"\n  id:      {obj['id']}\n"
                f"  file:    {obj.get('file_path', 'N/A')}\n"
                f"  updated: {obj.get('updated', 'N/A')}"
            )
            content = (obj.get("content") or "").strip()
            if content:
                preview = content[:400].replace("\n", " ")
                lines.append(f"  content: {preview}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching KS objects: {e}"


def get_ks_details_tool(slug: str) -> str:
    try:
        from api_client import get_ks_details

        data = get_ks_details(slug=slug)
        return str(data)
    except Exception as e:
        return f"Error: {e}"


def create_ks_tool(name: str, slug: str, description: str = "") -> str:
    try:
        from api_client import create_knowledge_source

        result = create_knowledge_source(name=name, slug=slug, description=description)
        return f"Created knowledge source: {result}"
    except Exception as e:
        return f"Error creating knowledge source: {e}"


def upload_to_ks_tool(file_path: str, ks_slug: str) -> str:
    fpath = _resolve(file_path)
    if not os.path.exists(fpath):
        return f"Error: file not found: {fpath}"
    try:
        from api_client import upload_file_to_ks

        result = upload_file_to_ks(local_path=fpath, ks_slug=ks_slug)
        return f"Upload complete: {result}"
    except Exception as e:
        return f"Error uploading to KS: {e}"
