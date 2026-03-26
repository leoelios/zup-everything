"""
Agent loop: sends prompts to the StackSpot chat API, parses tool calls
from the response, executes them, and loops until no more tool calls.
"""

import json
import os
import re
from typing import Callable, Generator, Optional

from ulid import ULID

import tools as tool_module

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Callable] = {
    "read_file": tool_module.read_file,
    "edit_file": tool_module.edit_file,
    "replace_lines": tool_module.replace_lines,
    "insert_after_line": tool_module.insert_after_line,
    "search_html": tool_module.search_html,
    "edit_html_attr": tool_module.edit_html_attr,
    "search_xml": tool_module.search_xml,
    "edit_xml_attr": tool_module.edit_xml_attr,
    "search_python": tool_module.search_python,
    "search_java": tool_module.search_java,
    "search_js": tool_module.search_js,
    "find_file": tool_module.find_file,
    "list_files": tool_module.list_files,
    "search_in_files": tool_module.search_in_files,
    "bash": tool_module.bash,
    "list_knowledge_sources": tool_module.list_knowledge_sources_tool,
    "get_ks_objects": tool_module.get_ks_objects_tool,
    "get_ks_details": tool_module.get_ks_details_tool,
    "create_knowledge_source": tool_module.create_ks_tool,
    "upload_to_knowledge_source": tool_module.upload_to_ks_tool,
    "web_search": tool_module.web_search,
    "fetch_page": tool_module.fetch_page,
    "ask_user": tool_module.ask_user,
}

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<name>([\w_]+)</name>\s*<parameters>(.*?)</parameters>\s*</tool_call>",
    re.DOTALL,
)
# Fallback: LLM sometimes emits <tool_call>toolname{...json...}</tool_call> (missing <name>/<parameters> tags)
TOOL_CALL_FALLBACK_RE = re.compile(
    r"<tool_call>\s*([\w_]+)\s*(\{.*?\})\s*(?:</parameters>)?\s*</tool_call>",
    re.DOTALL,
)
# Detect any <tool_call> tag (used to identify malformed calls)
TOOL_CALL_ANY_RE = re.compile(r"<tool_call>", re.DOTALL)
THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
TOOL_RESULT_RE = re.compile(r"<tool_result>.*?</tool_result>", re.DOTALL)

# Errors that should trigger self-correction
_ERROR_PREFIXES = ("Unknown tool", "Bad parameters", "Tool '", "Error", "error:")


def _is_bash_error(result: str) -> bool:
    """Detect non-zero exit codes from bash commands."""
    import re
    return bool(re.search(r"\[exit_code \d*[1-9]\d*\]", result))


_TAG_STRIP_RE = re.compile(r"<[^>]{1,40}>", re.DOTALL)


def get_activity_hint(content: str) -> str:
    """Call LLM to get a short phrase describing what the agent is currently doing."""
    from api_client import chat_nonstream
    from ulid import ULID as _ULID

    # Strip XML/tool tags so the hint LLM doesn't confuse them with HTML
    clean_content = THINKING_RE.sub("", content)
    clean_content = TOOL_CALL_RE.sub("", clean_content)
    clean_content = _TAG_STRIP_RE.sub("", clean_content).strip()

    # Nothing meaningful left after stripping — skip to avoid hallucinated hints
    if len(clean_content) < 30:
        return ""

    prompt = (
        "You are a progress tracker for an AI coding assistant. "
        "Analyze the snippet below and output ONE short phrase (4-10 words) describing the specific action being performed RIGHT NOW.\n\n"
        "Rules:\n"
        "- Be SPECIFIC: mention file names, function names, component names, error messages, or concepts involved.\n"
        "- Use active verbs: Implementando, Corrigindo, Refatorando, Adicionando, Removendo, Explicando, Verificando, Configurando, etc.\n"
        "- BAD (too generic): 'Analisando o código', 'Escrevendo código', 'Explicando erro'\n"
        "- GOOD (specific): 'Implementando autenticação JWT no middleware', 'Corrigindo NPE em UserService.findById', 'Adicionando validação no formulário de login'\n"
        "- Reply with ONLY the phrase — no punctuation at the end, no quotes, nothing else.\n"
        "- Use the same language as the content snippet.\n\n"
        f"Snippet:\n{clean_content[:800]}"
    )
    try:
        result = chat_nonstream(prompt, conversation_id=str(_ULID()))
        hint = result.get("message", "").strip()
        hint = THINKING_RE.sub("", hint).strip()
        return hint[:70] if hint else ""
    except Exception:
        return ""


def parse_thinking(text: str) -> str:
    """Extract content from <thinking>...</thinking> blocks."""
    blocks = THINKING_RE.findall(text)
    return "\n\n".join(b.strip() for b in blocks)


def strip_thinking(text: str) -> str:
    return THINKING_RE.sub("", text).strip()


# Expected parameter signatures shown to the LLM on correction
_TOOL_SIGNATURES: dict[str, str] = {
    "read_file":                 'read_file(path="<string>", start_line=<int optional>, end_line=<int optional>)',
    "find_file":                 'find_file(name="<glob e.g. index.html, *.py>", path="<optional dir>")',
    "edit_file":                 'edit_file(path="<string>", old_str="<exact text to replace>", new_str="<replacement>")',
    "replace_lines":             'replace_lines(path="<string>", start_line=<int>, end_line=<int>, new_content="<string>")',
    "insert_after_line":         'insert_after_line(path="<string>", line_number=<int>, new_content="<string>")',
    "search_html":               'search_html(path="<string>", selector="<css selector>")',
    "edit_html_attr":            'edit_html_attr(path="<string>", selector="<css selector>", attribute="<string>", value="<string>")',
    "search_xml":                'search_xml(path="<string>", xpath="<xpath expression>")',
    "edit_xml_attr":             'edit_xml_attr(path="<string>", xpath="<xpath expression>", attribute="<string>", value="<string>")',
    "search_python":             'search_python(path="<string>", name="<string>", kind="<function|class|import|any>")',
    "search_java":               'search_java(path="<string>", name="<string>", kind="<class|method|field|annotation|any>")',
    "search_js":                 'search_js(path="<string>", name="<string>", kind="<function|arrow|class|method|import|export|any>")',
    "list_files":                'list_files(path="<string optional>", pattern="<specific glob e.g. **/*.py>", max_depth=<int optional>)',
    "search_in_files":           'search_in_files(pattern="<regex or literal string>", path="<directory or file>", recursively=<true|false>)',
    "bash":                      'bash(command="<string>", timeout=<int optional>)',
    "list_knowledge_sources":    'list_knowledge_sources(page=<int optional>, size=<int optional>)',
    "get_ks_objects":            'get_ks_objects(slug="<string>", page=<int optional>, size=<int optional>)',
    "get_ks_details":            'get_ks_details(slug="<string>")',
    "create_knowledge_source":   'create_knowledge_source(name="<string>", slug="<string>", description="<optional>")',
    "upload_to_knowledge_source":'upload_to_knowledge_source(file_path="<string>", ks_slug="<string>")',
    "web_search":                'web_search(query="<string>", max_results=<int optional, default 6>)',
    "fetch_page":                'fetch_page(url="<string>", selector="<css selector optional>")',
    "ask_user":                  'ask_user(question="<string>", options=["<opt_a>", "<opt_b>", "<opt_c>"])',
}

_PARSE_ERROR_SENTINEL = "__PARSE_ERROR__"

# Fallback: model sometimes emits XML-style params like <path>foo</path> instead of JSON
_XML_PARAM_RE = re.compile(r"<([\w_]+)>(.*?)</\1>", re.DOTALL)


def _try_parse_xml_params(text: str) -> dict | None:
    """Try to parse XML-style parameters as a last resort, e.g. <path>foo</path>."""
    matches = _XML_PARAM_RE.findall(text)
    if not matches:
        return None
    return {k: v.strip() for k, v in matches}


def _try_parse_python_dict(text: str) -> dict | None:
    """Try to parse Python dict syntax, e.g. {'path': 'foo'} (single quotes)."""
    import ast
    try:
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass
    return None


def _parse_params(params_str: str) -> tuple[dict, str | None]:
    """Parse a JSON params string, with XML and Python-dict fallbacks. Returns (params, error)."""
    try:
        return json.loads(params_str), None
    except json.JSONDecodeError as e:
        xml_params = _try_parse_xml_params(params_str)
        if xml_params is not None:
            return xml_params, None
        py_params = _try_parse_python_dict(params_str)
        if py_params is not None:
            return py_params, None
        return {}, f"{e} — raw content was: {params_str!r}"


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        params, parse_error = _parse_params(m.group(2).strip())
        calls.append({"name": m.group(1).strip(), "parameters": params, "_parse_error": parse_error})

    # Fallback: try <tool_call>toolname{...json...}</tool_call> (missing <name>/<parameters> tags)
    if not calls and TOOL_CALL_ANY_RE.search(text):
        for m in TOOL_CALL_FALLBACK_RE.finditer(text):
            params, parse_error = _parse_params(m.group(2).strip())
            calls.append({"name": m.group(1).strip(), "parameters": params, "_parse_error": parse_error, "_format_fallback": True})

    return calls


def strip_tool_calls(text: str) -> str:
    text = TOOL_CALL_RE.sub("", text)
    text = TOOL_CALL_FALLBACK_RE.sub("", text)
    return text.strip()


def _is_error(result: str) -> bool:
    return any(result.startswith(p) for p in _ERROR_PREFIXES) or _is_bash_error(result)


def execute_tool(name: str, parameters: dict, parse_error: str | None = None, registry: dict | None = None) -> str:
    # JSON was malformed — tell the LLM exactly what it sent and the correct format
    if parse_error:
        sig = _TOOL_SIGNATURES.get(name, name)
        return (
            f"PARAMETER PARSE ERROR for tool '{name}': {parse_error}\n"
            f"The parameters block must be valid JSON. Correct signature:\n"
            f"  {sig}\n"
            f"Please retry with properly formatted JSON parameters."
        )

    fn = (registry if registry is not None else TOOL_REGISTRY).get(name)
    if fn is None:
        available = "\n".join(f"  {s}" for s in _TOOL_SIGNATURES.values())
        return (
            f"Unknown tool: '{name}'.\n"
            f"Available tools with signatures:\n{available}\n"
            f"Please retry using one of the exact tool names above."
        )
    try:
        return fn(**parameters)
    except TypeError as e:
        sig = _TOOL_SIGNATURES.get(name, name)
        return (
            f"Wrong parameters for tool '{name}': {e}\n"
            f"You passed: {json.dumps(parameters)}\n"
            f"Correct signature: {sig}\n"
            f"Please retry with the correct parameter names."
        )
    except Exception as e:
        return f"Tool '{name}' error: {e}\nPlease retry or use a different approach."


def _correction_note() -> str:
    """Appended when any tool result contains an error — prompts self-correction."""
    return (
        "\n<system_note>Tool error. Fix parameters and retry. "
        "For edit_file: read_file first to get exact text. "
        "For bash: check [exit_code] and [stderr], retry with corrected command.</system_note>"
    )


_READ_ONLY_TOOLS = {
    "read_file", "find_file", "list_files", "search_in_files",
    "search_html", "search_xml", "search_python", "search_java", "search_js",
    "list_knowledge_sources", "get_ks_objects", "get_ks_details",
    "web_search", "fetch_page",
}

_WRITE_TOOLS = {
    "edit_file", "replace_lines", "insert_after_line",
    "edit_html_attr", "edit_xml_attr",
}


def _completion_note(last_tools: list[str] | None = None) -> str:
    """Appended when all tool calls succeeded — model decides whether to continue or wrap up."""
    if last_tools and any(t in _WRITE_TOOLS for t in last_tools):
        return "\n<system_note>Write done. Verify with read_file, then respond.</system_note>"
    return "\n<system_note>Tool done. Continue or give final answer.</system_note>"


def _ask_user_note() -> str:
    """Appended after an ask_user tool result — allows the model to keep using tools."""
    return "\n<system_note>User answered. Continue.</system_note>"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are Zup CLI, an AI coding assistant. cwd: {cwd} (Windows, cmd.exe).

## Tool call format
<tool_call><name>TOOL_NAME</name><parameters>{{"key": "value"}}</parameters></tool_call>
One tool per turn. Parameters: valid JSON. After each <tool_result>, call the next tool or give your final answer.

## Rules
- Think in <thinking>...</thinking>, then immediately emit a <tool_call> — no prose between them.
- TOOL FIRST: anything involving files, code, or system state → call a tool. Never answer from memory about this codebase.
- Read before write: always read_file before editing.
- After editing: verify with read_file. For .py run `python -m py_compile`, for .js run `node --check`.
- Minimal edits: change only what is asked.
- Questions: use ask_user, never plain text.
- Bash: Windows commands. [exit_code N≠0] = failure, diagnose and retry.
- KNOWLEDGE SOURCE (KS) — proactive saving: whenever you learn something important (business rules, decisions, architecture, credentials format, API details, recurring patterns, user preferences, etc.), save it to the appropriate KS without being asked. Use list_knowledge_sources() to find the best-fit KS; if none fits, create a new one with create_knowledge_source before uploading.
- ALWAYS save important information to KS: write the content to a temp file first (bash or edit_file), then call upload_to_knowledge_source(file_path, ks_slug).
- AFTER every upload_to_knowledge_source: call get_ks_objects(slug=<ks_slug>) and confirm the uploaded object appears. If it does NOT appear, report the failure — never claim the save succeeded without this verification.
- WHEN user requests business or technical information you don't know: call list_knowledge_sources() to find the relevant KS, then search its objects for the answer.
- DO NOT use get_ks_objects or get_ks_details to browse KS out of curiosity — only use them for post-save verification or when explicitly searching for specific information.

## Tools
read_file(path, start_line?, end_line?)
edit_file(path, old_str, new_str) — old_str="" to create file
replace_lines(path, start_line, end_line, new_content) — use when edit_file fails (special chars/emojis)
insert_after_line(path, line_number, new_content)
find_file(name, path?)
list_files(path?, pattern?, max_depth?)
search_in_files(pattern, path, recursively?)
search_html(path, selector) / edit_html_attr(path, selector, attribute, value)
search_xml(path, xpath) / edit_xml_attr(path, xpath, attribute, value)
search_python(path, name, kind?) / search_java(path, name, kind?) / search_js(path, name, kind?)
bash(command, timeout?)
ask_user(question, options)
web_search(query, max_results?) / fetch_page(url, selector?)
list_knowledge_sources() / get_ks_objects(slug) / get_ks_details(slug)
create_knowledge_source(name, slug) / upload_to_knowledge_source(file_path, ks_slug)
"""


def build_system_prompt() -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(cwd=os.getcwd())


def _tool_reminder() -> str:
    return (
        f"cwd: {os.getcwd()} | "
        "Call a tool now — no prose. "
        'Format: <tool_call><name>TOOL_NAME</name><parameters>{"key": "value"}</parameters></tool_call>'
    )


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

_RESULT_CONTENT_RE = re.compile(r"<content>(.*?)</content>", re.DOTALL)


class Agent:
    MAX_TOOL_ITERATIONS = 30
    MAX_CONTINUATION_ITERATIONS = 15

    # Tools that mutate the filesystem or run shell commands — require user confirmation
    CONFIRM_TOOLS = {"edit_file", "replace_lines", "insert_after_line", "edit_html_attr", "edit_xml_attr", "bash"}

    def __init__(
        self,
        on_tool_use: Optional[Callable[[str, dict], None]] = None,
        on_tool_result: Optional[Callable[[str, str], None]] = None,
        on_thinking: Optional[Callable[[str], None]] = None,
        on_llm_chunk: Optional[Callable[[str], None]] = None,
        on_confirm_tool: Optional[Callable[[str, dict], bool]] = None,
        on_llm_start: Optional[Callable[[int], None]] = None,
        on_token_count: Optional[Callable[[int, int], None]] = None,
        on_llm_activity: Optional[Callable[[str], None]] = None,
        on_bash_output: Optional[Callable[[str, bool], None]] = None,
    ):
        from config import get_config
        self.conversation_id: str = str(ULID())
        self._initialized = False
        # Restore last-used model from config (persisted across sessions)
        cfg = get_config()
        self.selected_model: Optional[str] = cfg.get("selected_model_id")
        self.selected_model_name: Optional[str] = cfg.get("selected_model_name")
        self.selected_agent_id: Optional[str] = cfg.get("selected_agent_id")
        self.selected_agent_name: Optional[str] = cfg.get("selected_agent_name")
        self.on_tool_use = on_tool_use or (lambda n, p: None)
        self.on_tool_result = on_tool_result or (lambda n, r: None)
        self.on_thinking = on_thinking or (lambda t: None)
        self.on_llm_chunk = on_llm_chunk or (lambda c: None)
        # Returns True to allow, False to deny; defaults to always allow
        self.on_confirm_tool = on_confirm_tool or (lambda n, p: True)
        self.on_llm_start = on_llm_start or (lambda n: None)
        self.on_token_count = on_token_count or (lambda i, o: None)
        self.on_llm_activity = on_llm_activity or (lambda t: None)
        self.on_bash_output = on_bash_output or (lambda line, is_stderr: None)
        # Per-instance tool registry — isolated copy so parallel tasks don't share state
        self._tool_registry: dict = dict(TOOL_REGISTRY)
        # Conversation history: list of {"user": str, "assistant": str} dicts
        self._history: list[dict] = []
        # True at the start of each run() call; reset after the first _stream_collect
        self._start_of_new_turn: bool = True

    def set_model(self, model_id: str, model_name: str):
        """Set active model and persist the choice."""
        from config import get_config, save_config
        self.selected_model = model_id
        self.selected_model_name = model_name
        cfg = get_config()
        cfg["selected_model_id"] = model_id
        cfg["selected_model_name"] = model_name
        save_config(cfg)

    def set_agent(self, agent_id: str, agent_name: str):
        """Set active agent and persist the choice."""
        from config import get_config, save_config
        self.selected_agent_id = agent_id
        self.selected_agent_name = agent_name
        cfg = get_config()
        cfg["selected_agent_id"] = agent_id
        cfg["selected_agent_name"] = agent_name
        save_config(cfg)

    def reset(self):
        self.conversation_id = str(ULID())
        self._initialized = False
        # selected_model / selected_model_name intentionally preserved across /reset

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_first_prompt(self, user_message: str) -> str:
        base = build_system_prompt()
        if self._history:
            hist_lines = []
            for h in self._history[-8:]:
                hist_lines.append(f"User: {h['user']}")
                # No truncation — the last response must be fully available
                hist_lines.append(f"Assistant: {h['assistant']}")
            history_block = "\n".join(hist_lines)
            return f"{base}\n\n## Previous conversation\n{history_block}\n\n---\n\nUser request: {user_message}"
        return f"{base}\n\n---\n\nUser request: {user_message}"

    def _build_followup_prompt(self, user_message: str) -> str:
        return f"{_tool_reminder()}\n\n{user_message}"

    def _call_api(self, prompt: str, streaming: bool = False):
        from api_client import chat_nonstream, chat_stream

        full_prompt = (
            self._build_first_prompt(prompt) if not self._initialized
            else self._build_followup_prompt(prompt)
        )

        if streaming:
            return chat_stream(
                full_prompt,
                conversation_id=self.conversation_id,
                agent_id=self.selected_agent_id,
                selected_model=None #self.selected_model,
            )
        else:
            result = chat_nonstream(
                full_prompt,
                conversation_id=self.conversation_id,
                agent_id=self.selected_agent_id,
                selected_model=None #self.selected_model,
            )
            if not self._initialized:
                self._initialized = True
            return result

    def _extract_message(self, result: dict) -> str:
        return result.get("message", "")

    def _stream_collect(self, prompt: str) -> tuple[str, dict]:
        """
        Stream one LLM turn, calling on_llm_start / on_llm_chunk / on_token_count.
        Returns (full_message, token_info).
        """
        import logger
        from api_client import chat_stream

        full_prompt = (
            self._build_first_prompt(prompt) if not self._initialized
            else self._build_followup_prompt(prompt)
        )

        logger.log_api_request(full_prompt, self.conversation_id, self.selected_model, streaming=True)
        self.on_llm_start(len(full_prompt))

        full_message = ""
        token_info: dict = {}
        _HINT_INTERVAL = 600
        _last_hint_len = -_HINT_INTERVAL  # first fires as soon as enough content arrives

        for chunk in chat_stream(
            full_prompt,
            conversation_id=self.conversation_id,
            agent_id=self.selected_agent_id,
            selected_model=self.selected_model,
        ):
            # Extract tokens from ANY chunk that carries them
            for tkey in ("tokens", "token_usage", "usage"):
                raw_t = chunk.get(tkey)
                if isinstance(raw_t, dict):
                    in_t = raw_t.get("input") or raw_t.get("input_tokens") or raw_t.get("prompt_tokens") or 0
                    out_t = raw_t.get("output") or raw_t.get("output_tokens") or raw_t.get("completion_tokens") or 0
                    if in_t or out_t:
                        token_info = {"input": in_t, "output": out_t}
                    break

            msg = chunk.get("message", "")
            if msg:
                full_message += msg
                self.on_llm_chunk(msg)
                if (len(full_message) - _last_hint_len) >= _HINT_INTERVAL:
                    _last_hint_len = len(full_message)
                    thinking_so_far = parse_thinking(full_message)
                    self.on_llm_activity(thinking_so_far or full_message[max(0, _last_hint_len - _HINT_INTERVAL):])

            stop_val = chunk.get("finish_reason")
            if stop_val and stop_val != "tool_use":
                if not self._initialized:
                    self._initialized = True
                break

        self.on_token_count(token_info.get("input", 0), token_info.get("output", 0))
        logger.log_api_response({"message": full_message, "tokens": token_info, "conversation_id": self.conversation_id})
        return full_message, token_info

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _process_response(self, message: str) -> tuple[list[dict], str]:
        """
        Parse a raw LLM message:
        - Emit thinking via callback
        - Return (tool_calls, clean_text_without_thinking_or_tool_tags)
        """
        import logger
        thinking = parse_thinking(message)
        if thinking:
            self.on_thinking(thinking)
            logger.log_thinking(thinking)

        clean = strip_thinking(message)
        tool_calls = parse_tool_calls(clean)
        text_part = strip_tool_calls(clean)
        return tool_calls, text_part

    def _execute_tools(self, tool_calls: list[dict]) -> tuple[list[str], bool]:
        """
        Execute tool calls, emit callbacks.
        Returns (result_blocks, had_errors).
        """
        import logger
        parts = []
        had_errors = False
        for tc in tool_calls:
            self.on_tool_use(tc["name"], tc["parameters"])
            logger.log_tool_call(tc["name"], tc["parameters"])
            # For mutating tools, ask the user to confirm before executing
            if tc["name"] in self.CONFIRM_TOOLS and not tc.get("_parse_error"):
                allowed = self.on_confirm_tool(tc["name"], tc["parameters"])
                logger.log_tool_confirm(tc["name"], allowed)
                if allowed is not True:
                    if isinstance(allowed, str) and allowed:
                        llm_text = (
                            f"User declined the '{tc['name']}' action and provided a reason: \"{allowed}\". "
                            "Reconsider your approach taking this feedback into account. "
                            "If a different action or approach would satisfy the user's goal, use that instead. "
                            "If you need clarification, use the ask_user tool. "
                            "NEVER explain what you would do in plain text — always use tools."
                        )
                        display_text = f"Declined: {allowed}"
                    else:
                        llm_text = (
                            f"User declined the '{tc['name']}' action. "
                            "Continue working toward the user's goal using other tools. "
                            "If you need clarification or want to propose an alternative approach, use the ask_user tool. "
                            "NEVER explain what you would do in plain text — always use tools."
                        )
                        display_text = "Declined"
                    result_text = llm_text
                    self.on_tool_result(tc["name"], display_text)
                    logger.log_tool_result(tc["name"], llm_text)
                    parts.append(
                        f"<tool_result>\n"
                        f"<name>{tc['name']}</name>\n"
                        f"<content>{result_text}</content>\n"
                        f"</tool_result>"
                    )
                    continue
            if tc["name"] == "bash" and not tc.get("_parse_error"):
                import logger as _lg
                _lg.log_bash_start(tc["parameters"].get("command", ""))
                result_text = tool_module.bash(
                    **tc["parameters"],
                    on_output=lambda line, is_stderr: (
                        self.on_bash_output(line, is_stderr),
                        _lg.log_bash_output(line, is_stderr),
                    ),
                )
            else:
                result_text = execute_tool(
                    tc["name"],
                    tc["parameters"],
                    parse_error=tc.get("_parse_error"),
                    registry=self._tool_registry,
                )
            self.on_tool_result(tc["name"], result_text)
            logger.log_tool_result(tc["name"], result_text)
            # Truncate large results to avoid flooding the context
            _MAX_RESULT_CHARS = 12000
            truncated = result_text
            if len(result_text) > _MAX_RESULT_CHARS:
                cut = result_text[:_MAX_RESULT_CHARS]
                remaining = len(result_text) - _MAX_RESULT_CHARS
                hint = (
                    f"\n... [truncated — {remaining} more chars. "
                    f"Use read_file(path, start_line=N, end_line=M) to read specific line ranges. "
                    f"Do NOT re-read the full file — it will truncate the same way.]"
                ) if tc["name"] == "read_file" else (
                    f"\n... [truncated — {remaining} more chars. Narrow your search or use read_file with line ranges.]"
                )
                truncated = cut + hint
            parts.append(
                f"<tool_result>\n"
                f"<name>{tc['name']}</name>\n"
                f"<content>{truncated}</content>\n"
                f"</tool_result>"
            )
            if _is_error(result_text):
                had_errors = True
        return parts, had_errors

    def _agent_loop(
        self,
        original_request: str,
        initial_prompt: str,
        max_iterations: int,
        execution_log: list[str],
        context_summary: str = "",
    ) -> tuple[str | None, str, list[str]]:
        """
        Core agent loop. Runs up to max_iterations tool-call cycles.
        Returns (final_response | None, last_context_summary, updated_execution_log).
        None means the loop exhausted its iterations without a final answer.
        """
        prompt = initial_prompt

        for _ in range(max_iterations):
            message, _tokens = self._stream_collect(prompt)
            tool_calls, text_part = self._process_response(message)

            if not tool_calls:
                # Response was only <thinking> with no tool call and no text — force action
                if not strip_thinking(message).strip():
                    prompt = (
                        f"[Task: {original_request}]\n\n"
                        "ERROR: Your last response contained only a <thinking> block with no action.\n"
                        "You MUST either:\n"
                        "  1. Call a tool immediately with <tool_call>...\n"
                        "  2. OR provide your final answer as plain text if the task is complete.\n"
                        "Do one of these now."
                    )
                    continue
                # LLM hallucinated <tool_result> or emitted a completely unparseable <tool_call>
                has_tool_result = "<tool_result>" in message
                has_unparsed_call = TOOL_CALL_ANY_RE.search(message) is not None
                if has_tool_result or has_unparsed_call:
                    problem = (
                        "You generated <tool_result> tags yourself — that is reserved for the system."
                        if has_tool_result else
                        "You emitted a <tool_call> with an invalid format that could not be parsed."
                    )
                    correction = (
                        f"[Task: {original_request}]\n\n"
                        f"FORMAT ERROR: {problem}\n\n"
                        "The ONLY valid format to call a tool is:\n"
                        "<tool_call><name>TOOL_NAME</name>"
                        "<parameters>{\"key\": \"value\"}</parameters></tool_call>\n\n"
                        "Rules:\n"
                        "- <name> tag contains ONLY the tool name (e.g. read_file)\n"
                        "- <parameters> tag contains ONLY valid JSON\n"
                        "- Do NOT write the tool name before <name>, do NOT skip any tag\n"
                        "- <tool_result> is injected by the system — NEVER write it yourself\n\n"
                        "Retry now with the correct format."
                    )
                    prompt = correction
                    continue
                # Detect "narration instead of action" — agent explaining what it will do
                # instead of calling a tool. Only trigger when the task clearly needs more work.
                clean_text = TOOL_RESULT_RE.sub("", strip_thinking(message)).strip()
                _NARRATION_PHRASES = (
                    "i will ", "i'll ", "i'm going to", "i'm waiting", "let me ",
                    "once i have", "once i see", "i would ", "my plan", "i need to ",
                    "i should ", "next i will", "first i will",
                )
                lower_text = clean_text.lower()
                is_narration = (
                    any(lower_text.startswith(p) or f"\n{p}" in lower_text for p in _NARRATION_PHRASES)
                    and len(clean_text) < 800  # short responses without tool calls are suspect
                )
                if is_narration:
                    prompt = (
                        f"[Task: {original_request}]\n\n"
                        "VIOLATION: You wrote explanatory text instead of calling a tool.\n"
                        "Do NOT narrate, plan, or explain. Call a tool immediately.\n"
                        "Your next response must start with a <tool_call> block."
                    )
                    continue
                return clean_text, context_summary, execution_log

            # Execute only the FIRST tool call per iteration — forces step-by-step execution
            # and prevents the model from planning+executing everything at once.
            tool_calls = tool_calls[:1]
            result_parts, had_errors = self._execute_tools(tool_calls)

            # Record what was actually executed for continuation context
            for tc, rp in zip(tool_calls, result_parts):
                m = _RESULT_CONTENT_RE.search(rp)
                result_preview = (m.group(1).strip()[:200] if m else "")
                execution_log.append(f"• {tc['name']}: {result_preview}")

            tool_block = "\n\n".join(result_parts)
            used_ask_user = any(tc["name"] == "ask_user" for tc in tool_calls)
            if had_errors:
                suffix = _correction_note()
            elif used_ask_user:
                suffix = _ask_user_note()
            else:
                suffix = _completion_note([tc["name"] for tc in tool_calls])

            goal = f"[Task: {original_request}]"
            if text_part.strip():
                context_summary = text_part.strip()
            prompt = (
                f"{goal}\n\n[Progress]: {context_summary}\n\n{tool_block}{suffix}"
                if context_summary
                else f"{goal}\n\n{tool_block}{suffix}"
            )

        return None, context_summary, execution_log

    def run(self, user_message: str) -> str:
        """
        Agent loop with chain-of-thought and self-correction (streaming internally).
        If MAX_TOOL_ITERATIONS is hit, automatically continues with full execution context.
        Returns the final text response.
        """
        # Fresh conversation ID each run — history is injected manually, no server-side dependency
        self.conversation_id = str(ULID())
        self._initialized = False
        self._start_of_new_turn = False
        execution_log: list[str] = []

        result, context_summary, execution_log = self._agent_loop(
            original_request=user_message,
            initial_prompt=user_message,
            max_iterations=self.MAX_TOOL_ITERATIONS,
            execution_log=execution_log,
        )

        if result is not None:
            final = TOOL_RESULT_RE.sub("", result).strip()
            self._history.append({"user": user_message, "assistant": final})
            return final

        # Hit the iteration limit — build a continuation prompt with the real execution history
        done_summary = "\n".join(execution_log) if execution_log else "  (no tools executed)"
        continuation_prompt = (
            f"[Task: {user_message}]\n\n"
            f"<system_note>\n"
            f"You hit the iteration limit mid-task. Here is what was ACTUALLY executed so far:\n"
            f"{done_summary}\n\n"
            f"Your last recorded progress: {context_summary or '(none)'}\n\n"
            f"Resume the task from exactly where it stopped. Do NOT repeat anything already done above.\n"
            f"If the task is already complete based on what was done, provide your final response now.\n"
            f"</system_note>"
        )

        result, _, _ = self._agent_loop(
            original_request=user_message,
            initial_prompt=continuation_prompt,
            max_iterations=self.MAX_CONTINUATION_ITERATIONS,
            execution_log=execution_log,
            context_summary=context_summary,
        )

        if result is not None:
            final = TOOL_RESULT_RE.sub("", result).strip()
            self._history.append({"user": user_message, "assistant": final})
            return final
        final = (
            "Task incomplete after extended iterations. "
            "Review what was done and continue manually if needed."
        )
        self._history.append({"user": user_message, "assistant": final})
        return final

    def stream(self, user_message: str) -> Generator[str, None, None]:
        """Streaming-compatible agent loop (tool turns are non-streaming)."""
        original_request = user_message
        prompt = user_message
        stream_summary = ""

        for _ in range(self.MAX_TOOL_ITERATIONS):
            result = self._call_api(prompt, streaming=False)
            message = self._extract_message(result)

            tool_calls, text_part = self._process_response(message)

            if not tool_calls:
                yield strip_thinking(message)
                return

            tool_calls = tool_calls[:1]

            result_parts, had_errors = self._execute_tools(tool_calls)
            tool_block = "\n\n".join(result_parts)
            used_ask_user = any(tc["name"] == "ask_user" for tc in tool_calls)
            if had_errors:
                suffix = _correction_note()
            elif used_ask_user:
                suffix = _ask_user_note()
            else:
                suffix = _completion_note([tc["name"] for tc in tool_calls])

            goal = f"[Task: {original_request}]"
            if text_part.strip():
                stream_summary = text_part.strip()
            prompt = (
                f"{goal}\n\n[Progress]: {stream_summary}\n\n{tool_block}{suffix}"
                if stream_summary
                else f"{goal}\n\n{tool_block}{suffix}"
            )

        yield "Reached maximum tool iterations without a final response."
