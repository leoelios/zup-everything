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
    """Read a file with line numbers."""
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


_FULL_REWRITE_THRESHOLD = 0.6  # if old_str covers ≥60% of file lines, do a full rewrite


def edit_file(path: str, old_str: str, new_str: str) -> str:
    fpath = _resolve(path)
    # Allow creating new files by passing old_str=""
    if not os.path.exists(fpath):
        if old_str == "":
            parent = os.path.dirname(fpath)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(new_str)
            return f"Created {fpath}"
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

        if not new_content.strip():
            os.remove(fpath)
            return f"Deleted {fpath} (file became empty after edit)"

        # Heuristic: if the replaced chunk covers most of the file, skip the
        # string-search overhead and just overwrite the whole file directly.
        total_lines = max(len(content.splitlines()), 1)
        old_lines   = len(old_str.splitlines())
        if old_lines / total_lines >= _FULL_REWRITE_THRESHOLD:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(new_content)
            return f"Rewrote {fpath} (full overwrite — change spanned {old_lines}/{total_lines} lines)"

        with open(fpath, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"Edited {fpath}"
    except Exception as e:
        return f"Error editing file: {e}"


def list_files(path: str = ".", pattern: str = "**/*", max_depth: int = 3) -> str:
    """List files. Use specific patterns (e.g. '*.py', 'src/**/*.ts') to avoid huge outputs."""
    dpath = _resolve(path)
    if not os.path.exists(dpath):
        return f"Error: directory not found: {dpath}"

    # Warn and limit depth for broad patterns to avoid token waste
    is_broad = pattern in ("**/*", "**/.*", "*") or pattern == "**/*.*"
    if is_broad:
        hint = (
            "TIP: Pattern '**/*' is very broad. "
            "Use search_files(pattern='keyword') to find code, "
            "or list_files(pattern='*.py') for a specific file type.\n"
        )
    else:
        hint = ""

    try:
        matches = glob(os.path.join(dpath, pattern), recursive=True)
        entries = []
        for m in sorted(matches):
            rel = os.path.relpath(m, dpath)
            # Enforce max_depth for broad patterns
            if is_broad and rel.count(os.sep) >= max_depth:
                continue
            # Skip common noise dirs
            parts = rel.replace("\\", "/").split("/")
            skip = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"}
            if any(p in skip for p in parts):
                continue
            if os.path.isdir(m):
                entries.append(rel + "/")
            else:
                entries.append(rel)
        if not entries:
            return f"{hint}No entries found in {dpath} matching '{pattern}'"
        cap = 100 if is_broad else 300
        truncated = entries[:cap]
        more = len(entries) - cap
        suffix = f"\n... ({more} more — use a specific pattern or search_files to narrow down)" if more > 0 else ""
        return f"{hint}Files in {dpath}:\n" + "\n".join(truncated) + suffix
    except Exception as e:
        return f"Error listing files: {e}"


def find_file(name: str, path: str = ".") -> str:
    """Find files by name pattern (glob). Use this to locate files by name, not content."""
    dpath = _resolve(path)
    SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"}
    matches = []
    for root, dirs, files in os.walk(dpath):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in files:
            if fnmatch.fnmatch(fname.lower(), name.lower()):
                rel = os.path.relpath(os.path.join(root, fname), dpath).replace("\\", "/")
                matches.append(rel)
    if not matches:
        return f"No file matching '{name}' found in {dpath}"
    return f"Found {len(matches)} file(s) matching '{name}':\n" + "\n".join(matches[:50])


def search_files(pattern: str, path: str = ".", file_glob: str = "*") -> str:
    """Search file contents with a regex. Returns every matching line with its line number."""
    dpath = _resolve(path)
    SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"}

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        # Pattern is not valid regex — fall back to literal string search
        regex = re.compile(re.escape(pattern), re.IGNORECASE)

    file_hits: dict[str, list[str]] = {}
    total = 0
    MAX_TOTAL = 200
    MAX_FILES = 30

    # Normalise file_glob: if it contains path separators (e.g. "**/*.ts", "src/*.js")
    # extract just the filename portion so fnmatch works against bare filenames.
    _glob_basename = file_glob.replace("\\", "/").split("/")[-1] or "*"

    try:
        for root, dirs, files in os.walk(dpath):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for fname in files:
                if not fnmatch.fnmatch(fname, _glob_basename):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                except OSError:
                    continue
                hits: list[str] = []
                for i, line in enumerate(lines):
                    if regex.search(line):
                        hits.append(f"  {i+1}: {line.rstrip()}")
                        total += 1
                        if total >= MAX_TOTAL:
                            break
                if hits:
                    rel = os.path.relpath(fpath, dpath).replace("\\", "/")
                    file_hits[rel] = hits
                if len(file_hits) >= MAX_FILES or total >= MAX_TOTAL:
                    break
            if len(file_hits) >= MAX_FILES or total >= MAX_TOTAL:
                break
    except Exception as e:
        return f"Error searching files: {e}"

    if not file_hits:
        return f"No matches for '{pattern}' in {dpath}"

    out = [f"Matches for '{pattern}' ({total} hits in {len(file_hits)} files):"]
    for rel, hits in file_hits.items():
        out.append(f"\n{rel}:")
        out.extend(hits)
    if total >= MAX_TOTAL:
        out.append(f"\n... truncated at {MAX_TOTAL} matches. Narrow your search pattern.")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def bash(command: str, timeout: int = 60, on_output=None) -> str:
    """
    Execute a shell command, streaming output lines via on_output(line, is_stderr).
    on_output is called in real-time as lines arrive.
    """
    import threading

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd(),
        )
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def _read(stream, lines, is_stderr: bool):
            for raw in stream:
                line = raw.rstrip("\n")
                lines.append(line)
                if on_output:
                    on_output(line, is_stderr)

        t_out = threading.Thread(target=_read, args=(proc.stdout, stdout_lines, False), daemon=True)
        t_err = threading.Thread(target=_read, args=(proc.stderr, stderr_lines, True), daemon=True)
        t_out.start()
        t_err.start()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            t_out.join(2)
            t_err.join(2)
            return f"Command timed out after {timeout}s"

        t_out.join()
        t_err.join()

        parts = []
        if stdout_lines:
            parts.append("\n".join(stdout_lines).rstrip())
        if stderr_lines:
            parts.append(f"[stderr]\n{chr(10).join(stderr_lines).rstrip()}")
        if proc.returncode != 0:
            parts.append(f"[exit_code {proc.returncode}]")
        return "\n".join(parts) if parts else "(no output)"
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


# ---------------------------------------------------------------------------
# Web tools
# ---------------------------------------------------------------------------

_FETCH_CHAR_LIMIT = 12_000  # max chars returned from a single page
_LABELS = ["a", "b", "c", "d"]


def ask_user(question: str, options: list) -> str:
    """
    Show an interactive multiple-choice question to the user.
    options: up to 3 choices provided by the agent (a/b/c).
    The last option is always d) free-text input by the user.
    Returns the chosen text or the custom typed answer.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style as PTStyle

    _prefix_re = re.compile(r"^[a-zA-Z0-9]+[).]\s*")
    agent_opts = [_prefix_re.sub("", str(o), count=1) for o in list(options)[:3]]
    all_opts   = agent_opts + ["Other... (type freely)"]
    labels     = _LABELS[:len(all_opts)]

    state = {"index": 0, "result": None}

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(event):
        state["index"] = (state["index"] - 1) % len(all_opts)

    @kb.add("down")
    @kb.add("j")
    def _down(event):
        state["index"] = (state["index"] + 1) % len(all_opts)

    @kb.add("enter")
    def _select(event):
        state["result"] = state["index"]
        event.app.exit()

    # Letter shortcuts
    for i, lbl in enumerate(labels):
        def _make_handler(idx):
            def _handler(event):
                state["index"] = idx
                state["result"] = idx
                event.app.exit()
            return _handler
        kb.add(lbl)(_make_handler(i))

    @kb.add("c-c")
    @kb.add("escape")
    def _cancel(event):
        state["result"] = -1
        event.app.exit()

    def _get_text():
        lines = [("class:question", f"\n  {question}\n\n")]
        for i, (lbl, opt) in enumerate(zip(labels, all_opts)):
            selected = i == state["index"]
            marker = "  ● " if selected else "  ○ "
            style  = "class:selected" if selected else (
                "class:other" if i == len(all_opts) - 1 else "class:option"
            )
            lines.append((style, f"{marker}{lbl}) {opt}\n"))
        lines.append(("class:hint", "\n  ↑/↓ or a/b/c/d · Enter to confirm · Esc to cancel\n\n"))
        return lines

    style = PTStyle.from_dict({
        "question": "bold cyan",
        "selected": "bold white reverse",
        "option":   "",
        "other":    "dim",
        "hint":     "dim",
    })

    layout = Layout(HSplit([
        Window(content=FormattedTextControl(_get_text, focusable=True)),
    ]))

    app: Application = Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=False,
    )
    app.run()

    chosen = state["result"]

    if chosen == -1:
        return "User cancelled without answering."

    # Last option → free-text input
    if chosen == len(all_opts) - 1:
        from prompt_toolkit import prompt as pt_prompt
        answer = pt_prompt("  Your answer: ").strip()
        return f"d) {answer}" if answer else "User left the free-text answer blank."

    return f"{labels[chosen]}) {all_opts[chosen]}"


def web_search(query: str, max_results: int = 6) -> str:
    """Search the web using DuckDuckGo and return a list of results."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "Error: 'duckduckgo-search' is not installed. Run: pip install duckduckgo-search"

    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"Error searching the web: {e}"

    if not hits:
        return f"No results found for: {query}"

    lines = [f"Web search results for: {query}\n"]
    for i, h in enumerate(hits, 1):
        title = h.get("title", "")
        url   = h.get("href", "")
        body  = h.get("body", "")
        lines.append(f"{i}. {title}\n   {url}\n   {body}\n")
    return "\n".join(lines)


def fetch_page(url: str, selector: str = "") -> str:
    """Fetch a web page and return its readable text content."""
    try:
        import requests
    except ImportError:
        return "Error: 'requests' is not installed. Run: pip install requests"
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return "Error: 'beautifulsoup4' is not installed. Run: pip install beautifulsoup4"

    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; zup-cli/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return f"Error fetching {url}: {e}"

    try:
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise tags
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Optional CSS selector to scope the content
        if selector:
            node = soup.select_one(selector)
            if node is None:
                return f"Error: selector '{selector}' not found on {url}"
            text = node.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Collapse blank lines
        lines = [ln for ln in text.splitlines() if ln.strip()]
        text = "\n".join(lines)

        if len(text) > _FETCH_CHAR_LIMIT:
            text = text[:_FETCH_CHAR_LIMIT] + f"\n\n... (truncated, {len(text)} total chars)"

        return f"Page: {url}\n\n{text}"
    except Exception as e:
        return f"Error parsing {url}: {e}"
