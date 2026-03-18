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
    "find_file": tool_module.find_file,
    "list_files": tool_module.list_files,
    "search_files": tool_module.search_files,
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
THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)

# Errors that should trigger self-correction
_ERROR_PREFIXES = ("Unknown tool", "Bad parameters", "Tool '", "Error", "error:")


def _is_bash_error(result: str) -> bool:
    """Detect non-zero exit codes from bash commands."""
    import re
    return bool(re.search(r"\[exit_code \d*[1-9]\d*\]", result))


def get_activity_hint(content: str) -> str:
    """Call LLM to get a short phrase describing what the agent is currently doing."""
    from api_client import chat_nonstream
    from ulid import ULID as _ULID

    prompt = (
        "Analyze the following AI assistant output and respond with ONE short phrase "
        "(3-8 words) describing what the AI is doing right now. "
        "Examples: 'Lendo arquivos do projeto', 'Escrevendo testes', 'Analisando o código', "
        "'Planning the implementation', 'Searching for relevant files'. "
        "Reply with ONLY the phrase — no punctuation at the end, nothing else.\n\n"
        f"Output:\n{content[:600]}"
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
    "read_file":                 'read_file(path="<string>")',
    "find_file":                 'find_file(name="<glob e.g. index.html, *.py>", path="<optional dir>")',
    "edit_file":                 'edit_file(path="<string>", old_str="<exact text to replace>", new_str="<replacement>")',
    "list_files":                'list_files(path="<string optional>", pattern="<specific glob e.g. **/*.py>", max_depth=<int optional>)',
    "search_files":              'search_files(pattern="<regex>", path="<optional>", file_glob="<optional>", context_lines=<int optional default 1>)',
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


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        name = m.group(1).strip()
        params_str = m.group(2).strip()
        parse_error = None
        try:
            params = json.loads(params_str)
        except json.JSONDecodeError as e:
            # Attempt XML-style fallback
            xml_params = _try_parse_xml_params(params_str)
            if xml_params is not None:
                params = xml_params
            else:
                # Attempt Python-dict fallback (e.g. single-quoted keys/values)
                py_params = _try_parse_python_dict(params_str)
                if py_params is not None:
                    params = py_params
                else:
                    params = {}
                    parse_error = f"{e} — raw content was: {params_str!r}"
        calls.append({"name": name, "parameters": params, "_parse_error": parse_error})
    return calls


def strip_tool_calls(text: str) -> str:
    return TOOL_CALL_RE.sub("", text).strip()


def _is_error(result: str) -> bool:
    return any(result.startswith(p) for p in _ERROR_PREFIXES) or _is_bash_error(result)


def execute_tool(name: str, parameters: dict, parse_error: str | None = None) -> str:
    # JSON was malformed — tell the LLM exactly what it sent and the correct format
    if parse_error:
        sig = _TOOL_SIGNATURES.get(name, name)
        return (
            f"PARAMETER PARSE ERROR for tool '{name}': {parse_error}\n"
            f"The parameters block must be valid JSON. Correct signature:\n"
            f"  {sig}\n"
            f"Please retry with properly formatted JSON parameters."
        )

    fn = TOOL_REGISTRY.get(name)
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
        "\n<system_note>\n"
        "One or more tool calls above returned errors. Instructions:\n"
        "1. Read each <tool_result> error message carefully.\n"
        "2. Fix the tool name or parameters exactly as shown in the error.\n"
        "3. For edit_file errors: first call read_file to get the exact text, "
        "then use that exact text as old_str.\n"
        "   IMPORTANT: If read_file returned 'Error: file not found', do NOT call "
        "edit_file on that path — the file does not exist. Use list_files or "
        "search_files to find the correct path before editing.\n"
        "4. For bash errors: check [exit_code N] and [stderr] output. Reason about "
        "WHY the command failed (missing dependency, wrong path, permission, syntax error, etc.) "
        "and retry with a corrected command or a different approach.\n"
        "5. Retry only the failed calls — do not repeat successful ones.\n"
        "</system_note>"
    )


_READ_ONLY_TOOLS = {
    "read_file", "find_file", "list_files", "search_files",
    "list_knowledge_sources", "get_ks_objects", "get_ks_details",
    "web_search", "fetch_page",
}


def _completion_note(last_tools: list[str] | None = None) -> str:
    """Appended when all tool calls succeeded — model decides whether to continue or wrap up."""
    if last_tools and all(t in _READ_ONLY_TOOLS for t in last_tools):
        return (
            "\n<system_note>\n"
            "Tool call completed. This was a read/search operation — the task is NOT done yet.\n"
            "You MUST continue: call the next tool to make progress toward completing the user's request.\n"
            "Do NOT report results or summarize until you have actually applied all changes.\n"
            "</system_note>"
        )
    return (
        "\n<system_note>\n"
        "Tool call completed. "
        "If the task is fully done, provide your final response now. "
        "If more steps are needed, call the next tool.\n"
        "</system_note>"
    )


def _ask_user_note() -> str:
    """Appended after an ask_user tool result — allows the model to keep using tools."""
    return (
        "\n<system_note>\n"
        "The user has answered your question via ask_user. "
        "Process their answer and continue. "
        "If you need further clarification, call ask_user again. "
        "NEVER write follow-up questions or option lists as plain text — always use ask_user.\n"
        "</system_note>"
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are Zup CLI, an AI coding assistant. You help users with software engineering \
tasks by reading and writing files, executing shell commands, and managing knowledge sources.

## Reasoning

Before acting on any non-trivial request, reason through it inside <thinking>...</thinking> tags.
Use this as your private scratchpad: break down the problem, plan which tools to use and in what
order, and check your logic before committing to an approach.

IMPORTANT: Thinking is planning only — it does NOT execute anything. After the </thinking> block,
you MUST output the actual <tool_call> tags in your response body to execute tools.
Thinking about calling a tool is NOT the same as calling it. The tool call XML must appear
OUTSIDE the thinking block to be executed.

Example:
<thinking>
The user wants to add a new function. I should first read the file to understand the structure,
then edit the right location. Let me list files first to find the correct path.
</thinking>

## Tool call

To use a tool output EXACTLY this format (no extra whitespace inside tags):

<tool_call><name>TOOL_NAME</name><parameters>{{"param": "value"}}</parameters></tool_call>

CRITICAL: The <parameters> block MUST contain valid JSON — never XML, never plain text.
Correct:   <parameters>{{"path": "src/main.py"}}</parameters>
WRONG:     <parameters><path>src/main.py</path></parameters>
WRONG:     <parameters>src/main.py</parameters>

Rules:
- Always read a file before editing it.
- After receiving <tool_result> blocks, continue working or give your final answer.
- Do NOT repeat tool calls that already have results.
- If a tool result contains an error, read the error carefully, correct your approach, and retry.
- NEVER ask the user a question or present options as plain text. \
- ALWAYS use the ask_user tool when you need clarification or want to offer choices.

## File Edit Rules

1. **Read before editing** — always call `read_file` first. Never assume file contents.
2. **Use `edit_file` for existing files**
3. **Base every edit on the actual file content** — copy the exact `old_str` from what `read_file` returned.
5. **Use the right search tool**:
   - Know the filename? → `find_file(name="index.html")`
   - Looking for code/text inside files? → `search_files(pattern="keyword")`
   - Avoid `list_files(pattern="**/*")`

## Available Tools

read_file        – Read a file with line numbers.
  params: {{"path": "<string>"}}

edit_file        – Modify an existing file, or create a new one.
  To modify: {{"path": "<string>", "old_str": "<exact unique text to replace>", "new_str": "<replacement>"}}
  To create:  {{"path": "<string>", "old_str": "", "new_str": "<full file content>"}}
  Always copy old_str verbatim from read_file output. Chain multiple calls for multiple changes.

find_file        – Find files by name. Use this first when you know the filename.
  params: {{"name": "<glob e.g. index.html, *.py, config.*>", "path": "<string (optional)>"}}

list_files       – List files in a directory with a specific glob pattern.
  params: {{"path": "<string (optional)>", "pattern": "<glob (e.g. **/*.py, src/**/*.ts)>", "max_depth": <int (optional, default 3)>}}
  WARNING: avoid pattern="**/*" — use find_file or search_files instead.

search_files     – Search file contents with a regex. Returns matching lines WITH context. Use this first.
  params: {{"pattern": "<regex>", "path": "<string (optional)>", "file_glob": "<glob (optional)>", "context_lines": <int (optional, default 1)>}}

bash             – Execute a shell command.
  params: {{"command": "<string>", "timeout": <int (optional, default 60)>}}
  output markers: [stderr] precedes error output; [exit_code N] (N≠0) means the command failed.
  On failure: reason about the cause from [stderr]/[exit_code], then retry with a fix or try an alternative approach.

list_knowledge_sources – List available knowledge sources.
  params: {{"page": <int (optional)>, "size": <int (optional)>}}

get_ks_objects   – Get documents stored in a knowledge source.
  params: {{"slug": "<string>", "page": <int (optional)>, "size": <int (optional)>}}

get_ks_details   – Get metadata for a single knowledge source.
  params: {{"slug": "<string>"}}

create_knowledge_source – Create a new knowledge source.
  params: {{"name": "<string>", "slug": "<string>", "description": "<string (optional)>"}}

upload_to_knowledge_source – Upload a local file to a knowledge source.
  params: {{"file_path": "<string>", "ks_slug": "<string>"}}

web_search       – Search the web via DuckDuckGo and return ranked results.
  params: {{"query": "<string>", "max_results": <int (optional, default 6)>}}

fetch_page       – Fetch a URL and return its readable text content (crawling).
  params: {{"url": "<string>", "selector": "<css selector (optional)>"}}
  Use selector to scope to a specific element, e.g. "article" or "main".

ask_user         – Ask the user a clarifying question with up to 3 choices; the last option is always free-text.
  params: {{"question": "<string>", "options": ["<opt_a>", "<opt_b>", "<opt_c>"]}}
  MANDATORY: whenever you need to ask the user any question or present options, you MUST use this
  tool — NEVER write questions or option lists as plain text in your response.
  Returns the option chosen (e.g. "a) ...") or the user's typed answer (e.g. "d) ...").

## Context
Working directory: {cwd}

## Response Style
- Be concise and direct. Lead with action, not explanation.
- After completing tasks give a brief summary.
- Use markdown for code and formatted output.
"""


def build_system_prompt() -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(cwd=os.getcwd())


_TOOL_REMINDER = """\
[System reminder]
You are an AI coding assistant with direct tool access. You MUST call tools yourself to read, write, and edit files.
NEVER ask the user to run commands manually. NEVER explain what the user should do. Just do it.
To act, emit tool calls in this exact format:
<tool_call><name>TOOL_NAME</name><parameters>{"key": "value"}</parameters></tool_call>
Parameters must be valid JSON with double-quoted keys and string values.
[End system reminder]"""


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class Agent:
    MAX_TOOL_ITERATIONS = 15

    # Tools that mutate the filesystem or run shell commands — require user confirmation
    CONFIRM_TOOLS = {"edit_file", "bash"}

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
        return f"{build_system_prompt()}\n\n---\n\nUser request: {user_message}"

    def _build_followup_prompt(self, user_message: str) -> str:
        return f"{_TOOL_REMINDER}\n\n{user_message}"

    def _call_api(self, prompt: str, streaming: bool = False):
        from api_client import chat_nonstream, chat_stream

        is_first = not self._initialized
        full_prompt = (
            self._build_first_prompt(prompt) if is_first
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

        is_first = not self._initialized
        full_prompt = (
            self._build_first_prompt(prompt) if is_first
            else self._build_followup_prompt(prompt)
        )

        logger.log_api_request(full_prompt, self.conversation_id, self.selected_model, streaming=True)
        self.on_llm_start(len(full_prompt))

        full_message = ""
        token_info: dict = {}
        _hint_fired = False
        _HINT_THRESHOLD = 150

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
                if not _hint_fired and len(full_message) >= _HINT_THRESHOLD:
                    _hint_fired = True
                    self.on_llm_activity(full_message)

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
                if not allowed:
                    result_text = (
                        f"User declined the '{tc['name']}' action. "
                        "Continue working toward the user's goal using other tools. "
                        "If you need clarification or want to propose an alternative approach, use the ask_user tool. "
                        "NEVER explain what you would do in plain text — always use tools."
                    )
                    self.on_tool_result(tc["name"], result_text)
                    logger.log_tool_result(tc["name"], result_text)
                    parts.append(
                        f"<tool_result>\n"
                        f"<name>{tc['name']}</name>\n"
                        f"<content>{result_text}</content>\n"
                        f"</tool_result>"
                    )
                    continue
            result_text = execute_tool(
                tc["name"],
                tc["parameters"],
                parse_error=tc.get("_parse_error"),
            )
            self.on_tool_result(tc["name"], result_text)
            logger.log_tool_result(tc["name"], result_text)
            # Truncate large results to avoid flooding the context
            _MAX_RESULT_CHARS = 6000
            truncated = result_text
            if len(result_text) > _MAX_RESULT_CHARS:
                truncated = result_text[:_MAX_RESULT_CHARS] + f"\n... [truncated — {len(result_text) - _MAX_RESULT_CHARS} more chars]"
            parts.append(
                f"<tool_result>\n"
                f"<name>{tc['name']}</name>\n"
                f"<content>{truncated}</content>\n"
                f"</tool_result>"
            )
            if _is_error(result_text):
                had_errors = True
        return parts, had_errors

    def run(self, user_message: str) -> str:
        """
        Agent loop with chain-of-thought and self-correction (streaming internally).
        Returns the final text response.
        """
        original_request = user_message
        prompt = user_message
        context_summary = ""

        for _ in range(self.MAX_TOOL_ITERATIONS):
            message, _tokens = self._stream_collect(prompt)

            tool_calls, text_part = self._process_response(message)

            if not tool_calls:
                return strip_thinking(message)

            # Execute only the FIRST tool call per iteration — forces step-by-step execution
            # and prevents the model from planning+executing everything at once (hallucination risk).
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
            # Rolling context: keep model's latest synthesis but discard old raw tool results.
            # This prevents the prompt from growing unboundedly across iterations.
            if text_part.strip():
                context_summary = text_part.strip()
            prompt = (
                f"{goal}\n\n[Progress]: {context_summary}\n\n{tool_block}{suffix}"
                if context_summary
                else f"{goal}\n\n{tool_block}{suffix}"
            )

        return "Reached maximum tool iterations without a final response."

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
