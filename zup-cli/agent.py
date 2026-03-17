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
    "write_file": tool_module.write_file,
    "edit_file": tool_module.edit_file,
    "list_files": tool_module.list_files,
    "search_files": tool_module.search_files,
    "bash": tool_module.bash,
    "list_knowledge_sources": tool_module.list_knowledge_sources_tool,
    "get_ks_objects": tool_module.get_ks_objects_tool,
    "get_ks_details": tool_module.get_ks_details_tool,
    "create_knowledge_source": tool_module.create_ks_tool,
    "upload_to_knowledge_source": tool_module.upload_to_ks_tool,
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


def parse_thinking(text: str) -> str:
    """Extract content from <thinking>...</thinking> blocks."""
    blocks = THINKING_RE.findall(text)
    return "\n\n".join(b.strip() for b in blocks)


def strip_thinking(text: str) -> str:
    return THINKING_RE.sub("", text).strip()


# Expected parameter signatures shown to the LLM on correction
_TOOL_SIGNATURES: dict[str, str] = {
    "read_file":                 'read_file(path="<string>")',
    "write_file":                'write_file(path="<string>", content="<string>")',
    "edit_file":                 'edit_file(path="<string>", old_str="<exact text to replace>", new_str="<replacement>")',
    "list_files":                'list_files(path="<string optional>", pattern="<glob optional>")',
    "search_files":              'search_files(pattern="<regex>", path="<optional>", file_glob="<optional>")',
    "bash":                      'bash(command="<string>", timeout=<int optional>)',
    "list_knowledge_sources":    'list_knowledge_sources(page=<int optional>, size=<int optional>)',
    "get_ks_objects":            'get_ks_objects(slug="<string>", page=<int optional>, size=<int optional>)',
    "get_ks_details":            'get_ks_details(slug="<string>")',
    "create_knowledge_source":   'create_knowledge_source(name="<string>", slug="<string>", description="<optional>")',
    "upload_to_knowledge_source":'upload_to_knowledge_source(file_path="<string>", ks_slug="<string>")',
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


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        name = m.group(1).strip()
        params_str = m.group(2).strip()
        parse_error = None
        try:
            params = json.loads(params_str)
        except json.JSONDecodeError as e:
            # Attempt XML-style fallback before reporting an error
            xml_params = _try_parse_xml_params(params_str)
            if xml_params is not None:
                params = xml_params
            else:
                params = {}
                parse_error = f"{e} — raw content was: {params_str!r}"
        calls.append({"name": name, "parameters": params, "_parse_error": parse_error})
    return calls


def strip_tool_calls(text: str) -> str:
    return TOOL_CALL_RE.sub("", text).strip()


def _is_error(result: str) -> bool:
    return any(result.startswith(p) for p in _ERROR_PREFIXES)


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
        "4. Retry only the failed calls — do not repeat successful ones.\n"
        "</system_note>"
    )


def _completion_note() -> str:
    """Appended when all tool calls succeeded — tells the model to stop and summarise."""
    return (
        "\n<system_note>\n"
        "All tool calls completed successfully. "
        "Provide your final response to the user now. "
        "Do NOT call any more tools unless the user explicitly asks for additional work.\n"
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

Example:
<thinking>
The user wants to add a new function. I should first read the file to understand the structure,
then edit the right location. Let me list files first to find the correct path.
</thinking>

## Tool Protocol

To use a tool output EXACTLY this format (no extra whitespace inside tags):

<tool_call><name>TOOL_NAME</name><parameters>{{"param": "value"}}</parameters></tool_call>

CRITICAL: The <parameters> block MUST contain valid JSON — never XML, never plain text.
Correct:   <parameters>{{"path": "src/main.py"}}</parameters>
WRONG:     <parameters><path>src/main.py</path></parameters>
WRONG:     <parameters>src/main.py</parameters>

Rules:
- You may chain multiple tool calls in one response.
- Always read a file before editing it.
- After receiving <tool_result> blocks, continue working or give your final answer.
- Do NOT repeat tool calls that already have results.
- If a tool result contains an error, read the error carefully, correct your approach, and retry.

## Available Tools

read_file        – Read a file with line numbers.
  params: {{"path": "<string>"}}

write_file       – Create or overwrite a file.
  params: {{"path": "<string>", "content": "<string>"}}

edit_file        – Replace ONE unique occurrence of a string in a file.
  params: {{"path": "<string>", "old_str": "<string>", "new_str": "<string>"}}

list_files       – List files in a directory.
  params: {{"path": "<string (optional)>", "pattern": "<glob (optional, default: **/*)>"}}

search_files     – Search file contents with a regex.
  params: {{"pattern": "<regex>", "path": "<string (optional)>", "file_glob": "<glob (optional)>"}}

bash             – Execute a shell command.
  params: {{"command": "<string>", "timeout": <int (optional, default 60)>}}

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

## Context
Working directory: {cwd}

## Response Style
- Be concise and direct. Lead with action, not explanation.
- After completing tasks give a brief summary.
- Use markdown for code and formatted output.
"""


def build_system_prompt() -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(cwd=os.getcwd())


_TOOL_REMINDER = (
    "[Tool reminder: use <tool_call><name>NAME</name><parameters>{{...}}</parameters></tool_call>]"
)


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class Agent:
    MAX_TOOL_ITERATIONS = 15

    # Tools that mutate the filesystem or run shell commands — require user confirmation
    CONFIRM_TOOLS = {"write_file", "edit_file", "bash"}

    def __init__(
        self,
        on_tool_use: Optional[Callable[[str, dict], None]] = None,
        on_tool_result: Optional[Callable[[str, str], None]] = None,
        on_thinking: Optional[Callable[[str], None]] = None,
        on_llm_chunk: Optional[Callable[[str], None]] = None,
        on_confirm_tool: Optional[Callable[[str, dict], bool]] = None,
    ):
        from config import get_config
        self.conversation_id: str = str(ULID())
        self._initialized = False
        # Restore last-used model from config (persisted across sessions)
        cfg = get_config()
        self.selected_model: Optional[str] = cfg.get("selected_model_id")
        self.selected_model_name: Optional[str] = cfg.get("selected_model_name")
        self.on_tool_use = on_tool_use or (lambda n, p: None)
        self.on_tool_result = on_tool_result or (lambda n, r: None)
        self.on_thinking = on_thinking or (lambda t: None)
        self.on_llm_chunk = on_llm_chunk
        # Returns True to allow, False to deny; defaults to always allow
        self.on_confirm_tool = on_confirm_tool or (lambda n, p: True)

    def set_model(self, model_id: str, model_name: str):
        """Set active model and persist the choice."""
        from config import get_config, save_config
        self.selected_model = model_id
        self.selected_model_name = model_name
        cfg = get_config()
        cfg["selected_model_id"] = model_id
        cfg["selected_model_name"] = model_name
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
                selected_model=self.selected_model,
            )
        else:
            result = chat_nonstream(
                full_prompt,
                conversation_id=self.conversation_id,
                selected_model=self.selected_model,
            )
            if not self._initialized:
                self._initialized = True
            return result

    def _extract_message(self, result: dict) -> str:
        return result.get("message", "")

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
                        "Do not retry this operation unless the user explicitly asks."
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
            parts.append(
                f"<tool_result>\n"
                f"<name>{tc['name']}</name>\n"
                f"<content>{result_text}</content>\n"
                f"</tool_result>"
            )
            if _is_error(result_text):
                had_errors = True
        return parts, had_errors

    def run(self, user_message: str) -> str:
        """
        Agent loop with chain-of-thought and self-correction.
        Returns the final text response.
        """
        prompt = user_message

        for _ in range(self.MAX_TOOL_ITERATIONS):
            result = self._call_api(prompt, streaming=False)
            message = self._extract_message(result)

            tool_calls, text_part = self._process_response(message)

            if not tool_calls:
                return strip_thinking(message)

            result_parts, had_errors = self._execute_tools(tool_calls)
            tool_block = "\n\n".join(result_parts)

            if had_errors:
                suffix = _correction_note()
            else:
                suffix = _completion_note()

            prompt = (
                f"{text_part}\n\n{tool_block}{suffix}"
                if text_part
                else f"{tool_block}{suffix}"
            )

        return "Reached maximum tool iterations without a final response."

    def stream(self, user_message: str) -> Generator[str, None, None]:
        """Streaming-compatible agent loop (tool turns are non-streaming)."""
        prompt = user_message

        for _ in range(self.MAX_TOOL_ITERATIONS):
            result = self._call_api(prompt, streaming=False)
            message = self._extract_message(result)

            tool_calls, text_part = self._process_response(message)

            if not tool_calls:
                yield strip_thinking(message)
                return

            result_parts, had_errors = self._execute_tools(tool_calls)
            tool_block = "\n\n".join(result_parts)
            suffix = _correction_note() if had_errors else _completion_note()

            prompt = (
                f"{text_part}\n\n{tool_block}{suffix}"
                if text_part
                else f"{tool_block}{suffix}"
            )

        yield "Reached maximum tool iterations without a final response."
