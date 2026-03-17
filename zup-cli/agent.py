"""
Agent loop: sends prompts to the StackSpot chat API, parses tool calls
from the response, executes them, and loops until no more tool calls.
"""

import json
import os
import re
from typing import Callable, Generator, Optional

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
# Tool call parsing
# ---------------------------------------------------------------------------

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<name>([\w_]+)</name>\s*<parameters>(.*?)</parameters>\s*</tool_call>",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        name = m.group(1).strip()
        params_str = m.group(2).strip()
        try:
            params = json.loads(params_str)
        except json.JSONDecodeError:
            params = {}
        calls.append({"name": name, "parameters": params})
    return calls


def strip_tool_calls(text: str) -> str:
    return TOOL_CALL_RE.sub("", text).strip()


def execute_tool(name: str, parameters: dict) -> str:
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return f"Unknown tool: '{name}'. Available: {', '.join(TOOL_REGISTRY)}"
    try:
        return fn(**parameters)
    except TypeError as e:
        return f"Bad parameters for tool '{name}': {e}"
    except Exception as e:
        return f"Tool '{name}' error: {e}"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are Zup CLI, an AI coding assistant. You help users with software engineering \
tasks by reading and writing files, executing shell commands, and managing knowledge sources.

## Tool Protocol

To use a tool output EXACTLY this format (no extra whitespace inside tags):

<tool_call><name>TOOL_NAME</name><parameters>{{"param": "value"}}</parameters></tool_call>

Rules:
- You may chain multiple tool calls in one response.
- Always read a file before editing it.
- After receiving <tool_result> blocks, continue working or give your final answer.
- Do NOT repeat tool calls that already have results.

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

    def __init__(
        self,
        on_tool_use: Optional[Callable[[str, dict], None]] = None,
        on_tool_result: Optional[Callable[[str, str], None]] = None,
        on_llm_chunk: Optional[Callable[[str], None]] = None,
    ):
        from config import get_config
        self.conversation_id: Optional[str] = None
        self._initialized = False
        # Restore last-used model from config (persisted across sessions)
        cfg = get_config()
        self.selected_model: Optional[str] = cfg.get("selected_model_id")
        self.selected_model_name: Optional[str] = cfg.get("selected_model_name")
        self.on_tool_use = on_tool_use or (lambda n, p: None)
        self.on_tool_result = on_tool_result or (lambda n, r: None)
        self.on_llm_chunk = on_llm_chunk

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
        self.conversation_id = None
        self._initialized = False
        # selected_model / selected_model_name intentionally preserved across /clear

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
                self.conversation_id = result.get("conversation_id")
                self._initialized = True
            return result

    def _extract_message(self, result: dict) -> str:
        return result.get("message", "")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, user_message: str) -> str:
        """
        Non-streaming agent loop. Returns the final text response.
        Tool-use notifications are emitted via callbacks.
        """
        prompt = user_message

        for iteration in range(self.MAX_TOOL_ITERATIONS):
            result = self._call_api(prompt, streaming=False)
            message = self._extract_message(result)

            tool_calls = parse_tool_calls(message)

            if not tool_calls:
                # No tool calls — this is the final answer
                return message

            # Execute all tool calls in this response
            tool_results_parts = []
            for tc in tool_calls:
                self.on_tool_use(tc["name"], tc["parameters"])
                result_text = execute_tool(tc["name"], tc["parameters"])
                self.on_tool_result(tc["name"], result_text)
                tool_results_parts.append(
                    f"<tool_result>\n"
                    f"<name>{tc['name']}</name>\n"
                    f"<content>{result_text}</content>\n"
                    f"</tool_result>"
                )

            # Build next prompt: any text the LLM wrote + tool results
            text_part = strip_tool_calls(message)
            tool_block = "\n\n".join(tool_results_parts)
            prompt = (f"{text_part}\n\n{tool_block}" if text_part else tool_block)

        return "Reached maximum tool iterations without a final response."

    def stream(self, user_message: str) -> Generator[str, None, None]:
        """
        Streaming agent loop.
        - Tool-calling turns: non-streaming (accumulate full response, execute tools).
        - Final turn (no tool calls): streaming (yield chunks to caller).
        """
        prompt = user_message

        for iteration in range(self.MAX_TOOL_ITERATIONS):
            # Non-streaming: check for tool calls first
            result = self._call_api(prompt, streaming=False)
            message = self._extract_message(result)

            tool_calls = parse_tool_calls(message)

            if not tool_calls:
                # Final answer — now re-request with streaming for live output.
                # We use the already-received message and stream it simulated,
                # OR we re-send the prompt with streaming=True.
                # To avoid a double API call, just yield the text we have.
                yield message
                return

            # Execute tools
            tool_results_parts = []
            for tc in tool_calls:
                self.on_tool_use(tc["name"], tc["parameters"])
                result_text = execute_tool(tc["name"], tc["parameters"])
                self.on_tool_result(tc["name"], result_text)
                tool_results_parts.append(
                    f"<tool_result>\n"
                    f"<name>{tc['name']}</name>\n"
                    f"<content>{result_text}</content>\n"
                    f"</tool_result>"
                )

            text_part = strip_tool_calls(message)
            tool_block = "\n\n".join(tool_results_parts)
            prompt = (f"{text_part}\n\n{tool_block}" if text_part else tool_block)

        yield "Reached maximum tool iterations without a final response."
