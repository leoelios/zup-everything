"""
Local tool implementations — file system, shell, knowledge sources.
All functions return a string result (success or error message).
"""

import fnmatch
import os
import re
import subprocess
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve(path: str) -> str:
    """Return a normalised absolute path.
    On Windows, prepend \\?\\ to lift the 260-char MAX_PATH limit for all
    file-system operations (open, exists, makedirs, os.walk, etc.).
    """
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    path = os.path.normpath(path)
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        path = "\\\\?\\" + path
    return path


def _clean_path(path: str) -> str:
    """Strip the Windows long-path prefix for human-readable messages."""
    return path[4:] if path.startswith("\\\\?\\") else path


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------

def read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
    """Read a file with line numbers. Use start_line/end_line to read a specific range."""
    fpath = _resolve(path)
    if not os.path.exists(fpath):
        return f"Error: file not found: {_clean_path(fpath)}"
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        total = len(all_lines)
        s = max(1, start_line) - 1
        e = min(total, end_line) if end_line > 0 else total
        lines = all_lines[s:e]
        numbered = "".join(f"{s + i + 1:6}\t{line}" for i, line in enumerate(lines))
        header = f"File: {_clean_path(fpath)} ({total} lines total, showing {s+1}-{e})\n\n"
        if e < total:
            header += (
                f"[TRUNCATED — showing lines {s+1}-{e} of {total}. "
                f"To read more: read_file(path, start_line={e+1}, end_line={min(total, e+100)})]\n\n"
            )
        return header + numbered
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
        return f"Wrote {len(content)} bytes to {_clean_path(fpath)}"
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
            return f"Created {_clean_path(fpath)}"
        return f"Error: file not found: {_clean_path(fpath)}"
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        count = content.count(old_str)
        if count == 0:
            return f"Error: string not found in {_clean_path(fpath)}"
        if count > 1:
            return (
                f"Error: string found {count} times — provide more surrounding context "
                "to make it unique."
            )

        new_content = content.replace(old_str, new_str, 1)

        if not new_content.strip():
            os.remove(fpath)
            return f"Deleted {_clean_path(fpath)} (file became empty after edit)"

        # Heuristic: if the replaced chunk covers most of the file, skip the
        # string-search overhead and just overwrite the whole file directly.
        total_lines = max(len(content.splitlines()), 1)
        old_lines   = len(old_str.splitlines())
        if old_lines / total_lines >= _FULL_REWRITE_THRESHOLD:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(new_content)
            return f"Rewrote {_clean_path(fpath)} (full overwrite — change spanned {old_lines}/{total_lines} lines)"

        with open(fpath, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"Edited {_clean_path(fpath)}"
    except Exception as e:
        return f"Error editing file: {e}"


def replace_lines(path: str, start_line: int, end_line: int, new_content: str) -> str:
    """Replace lines [start_line, end_line] (1-indexed, inclusive) with new_content."""
    fpath = _resolve(path)
    if not os.path.exists(fpath):
        return f"Error: file not found: {_clean_path(fpath)}"
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        total = len(lines)
        if start_line < 1 or end_line < start_line or start_line > total:
            return f"Error: invalid line range {start_line}-{end_line} (file has {total} lines)"
        end_line = min(end_line, total)
        replacement = new_content if new_content.endswith("\n") else new_content + "\n"
        new_lines = lines[:start_line - 1] + [replacement] + lines[end_line:]
        with open(fpath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        return f"Replaced lines {start_line}-{end_line} in {_clean_path(fpath)}"
    except Exception as e:
        return f"Error: {e}"


def insert_after_line(path: str, line_number: int, new_content: str) -> str:
    """Insert new_content after line_number (1-indexed) in the file."""
    fpath = _resolve(path)
    if not os.path.exists(fpath):
        return f"Error: file not found: {_clean_path(fpath)}"
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        total = len(lines)
        if line_number < 0 or line_number > total:
            return f"Error: line_number {line_number} out of range (file has {total} lines)"
        insertion = new_content if new_content.endswith("\n") else new_content + "\n"
        new_lines = lines[:line_number] + [insertion] + lines[line_number:]
        with open(fpath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        return f"Inserted after line {line_number} in {_clean_path(fpath)}"
    except Exception as e:
        return f"Error: {e}"


def list_files(path: str = ".", pattern: str = "**/*", max_depth: int = 3) -> str:
    """List files. Use specific patterns (e.g. '*.py', 'src/**/*.ts') to avoid huge outputs."""
    dpath = _resolve(path)
    if not os.path.exists(dpath):
        return f"Error: directory not found: {_clean_path(dpath)}"

    # Warn and limit depth for broad patterns to avoid token waste
    is_broad = pattern in ("**/*", "**/.*", "*") or pattern == "**/*.*"
    hint = (
        "TIP: Pattern '**/*' is very broad. "
        "Use search_in_files(pattern='keyword') to find code, "
        "or list_files(pattern='*.py') for a specific file type.\n"
    ) if is_broad else ""

    SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"}

    try:
        # pathlib.Path.glob supports ** patterns and handles \\?\ long paths on Windows
        base = Path(dpath)
        entries = []
        for m in sorted(base.glob(pattern)):
            try:
                rel = m.relative_to(base).as_posix()
            except ValueError:
                continue
            parts = rel.split("/")
            # Skip noise dirs
            if any(p in SKIP_DIRS for p in parts):
                continue
            # Enforce max_depth for broad patterns
            if is_broad and len(parts) > max_depth:
                continue
            entries.append(rel + "/" if m.is_dir() else rel)

        if not entries:
            return f"{hint}No entries found in {_clean_path(dpath)} matching '{pattern}'"
        cap = 100 if is_broad else 300
        truncated = entries[:cap]
        more = len(entries) - cap
        suffix = (
            f"\n... ({more} more — use a specific pattern or search_in_files to narrow down)"
            if more > 0 else ""
        )
        return f"{hint}Files in {_clean_path(dpath)}:\n" + "\n".join(truncated) + suffix
    except Exception as e:
        return f"Error listing files: {e}"


def find_file(name: str, path: str = ".") -> str:
    """Find files recursively by name pattern (glob). Searches all subdirectories. Use this to locate files by name, not content."""
    dpath = _resolve(path)
    # Auto-add wildcards if no glob chars present so bare names like "routes" match "routes.ts"
    pattern = name if any(c in name for c in ("*", "?", "[")) else f"*{name}*"
    SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"}
    matches = []
    for root, dirs, files in os.walk(dpath):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in files:
            if fnmatch.fnmatch(fname.lower(), pattern.lower()):
                rel = os.path.relpath(os.path.join(root, fname), dpath).replace("\\", "/")
                # Strip any residual \\?\ from relpath on Windows
                if rel.startswith("\\\\?\\"):
                    rel = rel[4:]
                matches.append(rel)
    if not matches:
        return f"No file matching '{name}' found (searched recursively under: {_clean_path(dpath)})"
    return (
        f"Found {len(matches)} file(s) matching '{name}' (searched under: {_clean_path(dpath)}):\n"
        + "\n".join(matches[:100])
    )


def search_in_files(pattern: str, path: str = ".", recursively: bool = True) -> str:
    """Search file contents for a regex or literal string pattern.
    path: directory or file to search in.
    recursively: if True, walks all subdirectories; if False, searches only the top-level directory."""
    dpath = _resolve(path)

    # Compile regex; fall back to literal search if pattern is not valid regex
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)

    # Collect all files to search
    all_files = []
    if os.path.isfile(dpath):
        all_files.append(dpath)
    elif os.path.isdir(dpath):
        if recursively:
            for root, dirs, files in os.walk(dpath):
                dirs[:] = [
                    d for d in dirs
                    if not d.startswith(".") and d not in {"node_modules", "__pycache__", ".venv", "venv"}
                ]
                for fname in files:
                    all_files.append(os.path.join(root, fname))
        else:
            for fname in os.listdir(dpath):
                fpath = os.path.join(dpath, fname)
                if os.path.isfile(fpath):
                    all_files.append(fpath)
    else:
        return f"Error: path not found: {_clean_path(dpath)}"

    if not all_files:
        return f"No files found under {_clean_path(dpath)}"

    # Search each file
    results = []
    for fpath in all_files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception:
            try:
                with open(fpath, "r", encoding="latin-1") as f:
                    lines = f.readlines()
            except Exception:
                continue

        hits = []
        for i, line in enumerate(lines):
            if regex.search(line):
                hits.append(f"  {i + 1}: {line.rstrip()}")

        if hits:
            rel = os.path.relpath(fpath, dpath).replace("\\", "/")
            if rel.startswith("\\\\?\\"):
                rel = rel[4:]
            results.append((rel, hits))

    if not results:
        return f"No matches for '{pattern}' in {_clean_path(dpath)} (searched {len(all_files)} file(s))"

    out = [f"Matches for '{pattern}' ({sum(len(h) for _, h in results)} hits in {len(results)} files):"]
    for rel, hits in results:
        out.append(f"\n{rel}:")
        out.extend(hits)
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
            cwd=os.getcwd(),
        )
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def _read(stream, lines, is_stderr: bool):
            for raw in stream:
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
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
        return f"Error: file not found: {_clean_path(fpath)}"
    try:
        from api_client import upload_file_to_ks

        result = upload_file_to_ks(local_path=fpath, ks_slug=ks_slug)
        return f"Upload complete: {result}"
    except Exception as e:
        return f"Error uploading to KS: {e}"


# ---------------------------------------------------------------------------
# Language-aware search & edit tools
# ---------------------------------------------------------------------------

def _read_lines(fpath: str) -> list[str]:
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()


# ── HTML ────────────────────────────────────────────────────────────────────

def search_html(path: str, selector: str) -> str:
    """Find HTML elements by CSS selector. Returns matched elements with line numbers."""
    fpath = _resolve(path)
    if not os.path.exists(fpath):
        return f"Error: file not found: {_clean_path(fpath)}"
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return "Error: beautifulsoup4 not installed. Run: pip install beautifulsoup4"
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        soup = BeautifulSoup(content, "html.parser")
        elements = soup.select(selector)
        if not elements:
            return f"No elements matching '{selector}' in {_clean_path(fpath)}"
        out = [f"Found {len(elements)} element(s) matching '{selector}':"]
        for i, el in enumerate(elements[:20]):
            line_no = getattr(el, "sourceline", "?")
            snippet = str(el)
            first_line = snippet.split("\n")[0][:120]
            out.append(f"\n  [{i+1}] line {line_no}: {first_line}")
        return "\n".join(out)
    except Exception as e:
        return f"Error: {e}"


def edit_html_attr(path: str, selector: str, attribute: str, value: str) -> str:
    """Set an attribute on every HTML element matching selector. Safe — only touches the attribute value."""
    fpath = _resolve(path)
    if not os.path.exists(fpath):
        return f"Error: file not found: {_clean_path(fpath)}"
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return "Error: beautifulsoup4 not installed."
    try:
        lines = _read_lines(fpath)
        content = "".join(lines)
        soup = BeautifulSoup(content, "html.parser")
        elements = soup.select(selector)
        if not elements:
            return f"No elements matching '{selector}' found in {_clean_path(fpath)}"
        changed = 0
        for el in elements:
            line_no = getattr(el, "sourceline", None)
            if line_no is None:
                continue
            old_line = lines[line_no - 1]
            attr_re = re.compile(rf'\b{re.escape(attribute)}=["\'][^"\']*["\']')
            if attr_re.search(old_line):
                new_line = attr_re.sub(f'{attribute}="{value}"', old_line, count=1)
            elif f"<{el.name}" in old_line:
                new_line = old_line.replace(f"<{el.name}", f'<{el.name} {attribute}="{value}"', 1)
            else:
                continue
            lines[line_no - 1] = new_line
            changed += 1
        if not changed:
            return (
                f"Could not patch attribute '{attribute}' — elements found but line numbers unavailable. "
                "Use replace_lines instead."
            )
        with open(fpath, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return f"Set {attribute}=\"{value}\" on {changed} element(s) matching '{selector}' in {_clean_path(fpath)}"
    except Exception as e:
        return f"Error: {e}"


# ── XML ─────────────────────────────────────────────────────────────────────

def search_xml(path: str, xpath: str) -> str:
    """Find XML elements by XPath expression. Returns tag, attributes, text, and line numbers."""
    fpath = _resolve(path)
    if not os.path.exists(fpath):
        return f"Error: file not found: {_clean_path(fpath)}"
    try:
        import xml.etree.ElementTree as ET
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            raw_lines = f.readlines()
        tree = ET.parse(fpath)
        root = tree.getroot()
        elements = root.findall(xpath)
        if not elements:
            return f"No elements matching XPath '{xpath}' in {_clean_path(fpath)}"
        out = [f"Found {len(elements)} element(s) matching '{xpath}':"]
        for i, el in enumerate(elements[:20]):
            tag_local = el.tag.split("}")[-1] if "}" in el.tag else el.tag
            attrs = " ".join(f'{k}="{v}"' for k, v in el.attrib.items())
            text = (el.text or "").strip()[:80]
            pattern = re.compile(rf"<{re.escape(tag_local)}[\s>]")
            hit_line = next(
                (ln + 1 for ln, line in enumerate(raw_lines) if pattern.search(line)),
                "?"
            )
            out.append(f"\n  [{i+1}] line {hit_line}: <{tag_local} {attrs}> text={text!r}")
        return "\n".join(out)
    except Exception as e:
        return f"Error: {e}"


def edit_xml_attr(path: str, xpath: str, attribute: str, value: str) -> str:
    """Set an attribute on XML elements matching XPath. Writes back preserving structure."""
    fpath = _resolve(path)
    if not os.path.exists(fpath):
        return f"Error: file not found: {_clean_path(fpath)}"
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(fpath)
        root = tree.getroot()
        elements = root.findall(xpath)
        if not elements:
            return f"No elements matching XPath '{xpath}'"
        for el in elements:
            el.set(attribute, value)
        tree.write(fpath, encoding="unicode", xml_declaration=False)
        return f"Set {attribute}=\"{value}\" on {len(elements)} element(s) in {_clean_path(fpath)}"
    except Exception as e:
        return f"Error: {e}"


# ── Python ───────────────────────────────────────────────────────────────────

def search_python(path: str, name: str, kind: str = "any") -> str:
    """
    Find Python definitions by name using AST.
    kind: 'function', 'class', 'import', or 'any'
    Returns definition with start/end line numbers for use with replace_lines.
    """
    fpath = _resolve(path)
    if not os.path.exists(fpath):
        return f"Error: file not found: {_clean_path(fpath)}"
    try:
        import ast as _ast
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
            raw_lines = source.splitlines()
        tree = _ast.parse(source, filename=_clean_path(fpath))
        results = []
        for node in _ast.walk(tree):
            match kind:
                case "function":
                    if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                        continue
                case "class":
                    if not isinstance(node, _ast.ClassDef):
                        continue
                case "import":
                    if not isinstance(node, (_ast.Import, _ast.ImportFrom)):
                        continue
                case _:
                    if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                                             _ast.ClassDef, _ast.Import, _ast.ImportFrom)):
                        continue
            node_name = getattr(node, "name", None)
            if node_name is None:
                # imports
                names = [a.name for a in getattr(node, "names", [])]
                module = getattr(node, "module", "")
                node_name = module + "." + ",".join(names) if module else ",".join(names)
            if name.lower() not in node_name.lower():
                continue
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            snippet = "\n".join(raw_lines[start - 1: min(start + 5, end)])
            results.append(f"  lines {start}-{end}: {snippet[:200]}")
        if not results:
            return f"No {kind} named '{name}' found in {_clean_path(fpath)}"
        return f"Found {len(results)} match(es) for '{name}' in {_clean_path(fpath)}:\n" + "\n".join(results)
    except SyntaxError as e:
        return f"Syntax error parsing {_clean_path(fpath)}: {e}"
    except Exception as e:
        return f"Error: {e}"


# ── Java ─────────────────────────────────────────────────────────────────────

_JAVA_PATTERNS: dict[str, str] = {
    "class":      r"(?:public|private|protected|abstract|final|\s)*\s+(?:class|interface|enum)\s+{name}\b",
    "method":     r"(?:public|private|protected|static|final|synchronized|\s)+[\w<>\[\]]+\s+{name}\s*\(",
    "field":      r"(?:public|private|protected|static|final|\s)+[\w<>\[\]]+\s+{name}\s*[=;]",
    "annotation": r"@{name}\b",
}

def search_java(path: str, name: str, kind: str = "any") -> str:
    """
    Find Java class/method/field/annotation by name.
    kind: 'class', 'method', 'field', 'annotation', or 'any'
    Returns line numbers for use with replace_lines.
    """
    fpath = _resolve(path)
    if os.path.isdir(fpath):
        # Search across all .java files in directory
        _SKIP = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"}
        results = []
        for root, dirs, files in os.walk(fpath):
            dirs[:] = [d for d in dirs if d not in _SKIP and not d.startswith(".")]
            for fname in files:
                if fname.endswith(".java"):
                    r = search_java(os.path.join(root, fname), name, kind)
                    if "No match" not in r and "Error" not in r:
                        results.append(r)
        return "\n\n".join(results) if results else f"No {kind} named '{name}' found in {_clean_path(fpath)}"
    if not os.path.exists(fpath):
        return f"Error: file not found: {_clean_path(fpath)}"
    try:
        raw_lines = _read_lines(fpath)
        patterns = (
            {kind: _JAVA_PATTERNS[kind]} if kind in _JAVA_PATTERNS
            else _JAVA_PATTERNS
        )
        hits = []
        for k, pat in patterns.items():
            regex = re.compile(pat.replace("{name}", re.escape(name)), re.IGNORECASE)
            for i, line in enumerate(raw_lines):
                if regex.search(line):
                    end = i
                    if "{" in line:
                        depth = line.count("{") - line.count("}")
                        j = i + 1
                        while j < len(raw_lines) and depth > 0:
                            depth += raw_lines[j].count("{") - raw_lines[j].count("}")
                            j += 1
                        end = j - 1
                    hits.append(f"  [{k}] lines {i+1}-{end+1}: {line.rstrip()}")
        if not hits:
            return f"No {kind} named '{name}' found in {_clean_path(fpath)}"
        return f"Found {len(hits)} match(es) for '{name}' in {_clean_path(fpath)}:\n" + "\n".join(hits)
    except Exception as e:
        return f"Error: {e}"


# ── JavaScript / TypeScript ──────────────────────────────────────────────────

_JS_PATTERNS: dict[str, str] = {
    "function":  r"(?:export\s+)?(?:async\s+)?function\s+{name}\s*\(",
    "arrow":     r"(?:export\s+)?(?:const|let|var)\s+{name}\s*=\s*(?:async\s*)?\(",
    "class":     r"(?:export\s+)?class\s+{name}\b",
    "method":    r"(?:async\s+)?{name}\s*\([^)]*\)\s*\{{",
    "import":    r"import\s+.*\b{name}\b.*from",
    "export":    r"export\s+(?:default\s+)?(?:const|let|var|function|class)\s+{name}\b",
}

def search_js(path: str, name: str, kind: str = "any") -> str:
    """
    Find JavaScript/TypeScript function/class/arrow/import by name.
    kind: 'function', 'arrow', 'class', 'method', 'import', 'export', or 'any'
    Returns line numbers for use with replace_lines.
    """
    fpath = _resolve(path)
    if os.path.isdir(fpath):
        exts = {".js", ".ts", ".jsx", ".tsx", ".mjs"}
        _SKIP = {"node_modules", ".git", "dist", "build", "__pycache__", ".next"}
        results = []
        for root, dirs, files in os.walk(fpath):
            dirs[:] = [d for d in dirs if d not in _SKIP and not d.startswith(".")]
            for fname in files:
                if any(fname.endswith(e) for e in exts):
                    r = search_js(os.path.join(root, fname), name, kind)
                    if "No match" not in r and "Error" not in r:
                        results.append(r)
        return "\n\n".join(results) if results else f"No {kind} named '{name}' found in {_clean_path(fpath)}"
    if not os.path.exists(fpath):
        return f"Error: file not found: {_clean_path(fpath)}"
    try:
        raw_lines = _read_lines(fpath)
        patterns = (
            {kind: _JS_PATTERNS[kind]} if kind in _JS_PATTERNS
            else _JS_PATTERNS
        )
        hits = []
        seen_lines: set[int] = set()
        for k, pat in patterns.items():
            regex = re.compile(pat.replace("{name}", re.escape(name)))
            for i, line in enumerate(raw_lines):
                if i in seen_lines:
                    continue
                if regex.search(line):
                    end = i
                    if "{" in line:
                        depth = line.count("{") - line.count("}")
                        j = i + 1
                        while j < len(raw_lines) and depth > 0:
                            depth += raw_lines[j].count("{") - raw_lines[j].count("}")
                            j += 1
                        end = j - 1
                    seen_lines.add(i)
                    hits.append(f"  [{k}] lines {i+1}-{end+1}: {line.rstrip()}")
        if not hits:
            return f"No {kind} named '{name}' found in {_clean_path(fpath)}"
        return f"Found {len(hits)} match(es) for '{name}' in {_clean_path(fpath)}:\n" + "\n".join(hits)
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Web tools
# ---------------------------------------------------------------------------

_FETCH_CHAR_LIMIT = 12_000  # max chars returned from a single page
_LABELS = ["a", "b", "c", "d"]


def _get_proxies() -> dict | None:
    """Read proxy settings from environment variables."""
    http  = os.environ.get("HTTP_PROXY")  or os.environ.get("http_proxy")
    https = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if not http and not https:
        return None
    proxies: dict[str, str] = {}
    if http:
        proxies["http"] = http
    if https:
        proxies["https"] = https
    return proxies


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
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "Error: 'ddgs' is not installed. Run: pip install ddgs"

    proxies = _get_proxies()
    proxy = (proxies or {}).get("https") or (proxies or {}).get("http")
    try:
        try:
            with DDGS(proxy=proxy) as ddgs:
                hits = list(ddgs.text(query, max_results=max_results))
        except TypeError:
            with DDGS(proxies=proxies) as ddgs:
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
        resp = requests.get(url, headers=headers, timeout=15, proxies=_get_proxies())
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
