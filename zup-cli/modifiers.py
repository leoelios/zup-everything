"""
Modifier system — @modifier prefix handling for user prompts.

Modifiers start with @ and can appear anywhere in the prompt.
Example: "@multi analyze all Python files in this project"
"""

import json
import re
import concurrent.futures
from typing import Optional

# Matches @word tokens preceded by whitespace or start-of-string (not inside words/emails)
MODIFIER_RE = re.compile(r'(?<!\S)@(\w+)\b', re.IGNORECASE)


def extract_modifiers(prompt: str) -> tuple[list[str], str]:
    """
    Extract @modifier tokens from a prompt.
    Tokens inside double-quoted strings are ignored (e.g. "user@host.com").
    A valid modifier must be preceded by whitespace or appear at the start.
    Returns (list_of_modifier_names_lowercased, prompt_with_modifiers_removed).
    """
    found: list[str] = []

    def _collect(m: re.Match) -> str:
        found.append(m.group(1).lower())
        return ''

    # Split on double-quoted strings; only process unquoted segments.
    # re.split with a capturing group keeps the quoted parts in the result at odd indices.
    segments = re.split(r'("(?:[^"\\]|\\.)*")', prompt)
    clean_parts = []
    for i, segment in enumerate(segments):
        if i % 2 == 1:      # inside quotes — leave untouched
            clean_parts.append(segment)
        else:               # outside quotes — extract modifiers
            clean_parts.append(MODIFIER_RE.sub(_collect, segment))

    clean = ''.join(clean_parts).strip()
    return found, clean


# ---------------------------------------------------------------------------
# @multi modifier
# ---------------------------------------------------------------------------

_DECOMPOSE_SYSTEM = """\
You are a task decomposition assistant. Your only job is to split a user request \
into independent parallel subtasks.

Rules:
- Output ONLY a valid JSON array of strings — no explanation, no markdown, no code fences.
- Each string is a self-contained, actionable subtask for a separate AI coding agent.
- Produce between 2 and 5 subtasks. More subtasks = more parallelism, but keep them meaningful.
- If the request cannot be meaningfully split, return a single-element array.

Example output:
["Read and summarize auth.py", "Read and summarize tools.py", "Read and summarize agent.py"]
"""

_SYNTHESIZE_SYSTEM = """\
You are a synthesis assistant. Multiple parallel AI agents each completed a portion \
of a larger task. Your job is to merge their outputs into one cohesive, well-structured \
final response.

Rules:
- Merge overlapping information without duplication.
- Resolve any conflicts by noting them explicitly.
- Keep the response concise and directly useful to the user.
- Use markdown headings/lists where appropriate.
"""


def _decompose_task(task: str, agent) -> list[str]:
    """Ask the LLM to split *task* into parallel subtasks. Returns a list of strings."""
    from api_client import chat_nonstream

    prompt = (
        f"{_DECOMPOSE_SYSTEM}\n\n"
        f"User request to decompose:\n\"{task}\""
    )
    try:
        result = chat_nonstream(
            prompt,
            conversation_id=agent.conversation_id + "-multi-decompose",
            agent_id=agent.selected_agent_id,
            selected_model=agent.selected_model,
        )
        raw = result.get("message", "").strip()
        # Strip accidental markdown code fences
        raw = re.sub(r'^```[a-z]*\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
        subtasks = json.loads(raw.strip())
        if isinstance(subtasks, list) and subtasks:
            return [str(s).strip() for s in subtasks[:5] if str(s).strip()]
    except Exception:
        pass
    # Fallback: treat the full task as a single subtask
    return [task]


def _run_subtask(task: str, agent) -> str:
    """Run *task* on a fresh, silent Agent instance. Returns the final response text."""
    from agent import Agent

    worker = Agent()  # fresh conversation, no UI callbacks
    worker.selected_model = agent.selected_model
    worker.selected_model_name = agent.selected_model_name
    worker.selected_agent_id = agent.selected_agent_id
    worker.selected_agent_name = agent.selected_agent_name
    try:
        return worker.run(task)
    except Exception as e:
        return f"[Subtask error: {e}]"


def _synthesize_results(original_task: str, subtasks: list[str], results: list[str], agent) -> str:
    """Ask the LLM to merge all subtask results into one final response."""
    from api_client import chat_nonstream

    numbered = "\n\n".join(
        f"### Agent {i + 1} — {subtasks[i]}\n\n{result}"
        for i, result in enumerate(results)
    )
    prompt = (
        f"{_SYNTHESIZE_SYSTEM}\n\n"
        f"Original user request:\n\"{original_task}\"\n\n"
        f"Parallel agent results:\n\n{numbered}"
    )
    try:
        result = chat_nonstream(
            prompt,
            conversation_id=agent.conversation_id + "-multi-synthesize",
            agent_id=agent.selected_agent_id,
            selected_model=agent.selected_model,
        )
        return result.get("message", "").strip()
    except Exception:
        # Graceful fallback: return concatenated results
        return numbered


def run_multi(prompt: str, agent) -> str:
    """
    @multi modifier entry point.

    Pipeline:
      1. Decompose — one LLM call splits the task into N independent subtasks.
      2. Execute   — N agent workers run the subtasks concurrently in a thread pool.
      3. Synthesize — one LLM call merges all results into a final response.
    """
    import display

    # ── Step 1: Decompose ────────────────────────────────────────────────────
    display.print_info("[@multi] Decomposing task into parallel subtasks…")
    subtasks = _decompose_task(prompt, agent)

    display.print_info(f"[@multi] Launching {len(subtasks)} parallel agent(s):")
    for i, st in enumerate(subtasks, 1):
        display.print_info(f"  [{i}] {st}")

    # ── Step 2: Execute in parallel ──────────────────────────────────────────
    results: list[str] = [""] * len(subtasks)
    completed = [0]

    def _worker(idx: int, task: str) -> tuple[int, str]:
        return idx, _run_subtask(task, agent)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(subtasks)) as pool:
        futures = {
            pool.submit(_worker, i, task): i
            for i, task in enumerate(subtasks)
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
            except Exception as e:
                idx = futures[future]
                results[idx] = f"[Error: {e}]"
            completed[0] += 1
            display.print_info(
                f"[@multi] Agent {futures[future] + 1}/{len(subtasks)} finished "
                f"({completed[0]}/{len(subtasks)} done)."
            )

    # ── Step 3: Synthesize ───────────────────────────────────────────────────
    if len(subtasks) == 1:
        return results[0]

    display.print_info("[@multi] Synthesizing results from all agents…")
    return _synthesize_results(prompt, subtasks, results, agent)


# ---------------------------------------------------------------------------
# @auto modifier
# ---------------------------------------------------------------------------

_AUTO_ORCHESTRATOR_SYSTEM = """\
You are an autonomous task orchestrator controlling an AI coding assistant.
The assistant needs your guidance to complete a task without any user input.

Given the original task, recent progress, and a question from the agent, respond
with the best answer to move the task forward.

Rules:
- Be decisive — pick the best option for completing the original task.
- If given a list of options, choose one by its exact text.
- If asked for free-form input, provide a clear and concise answer.
- Never ask for clarification — always make the best decision.
- Keep your answer short and actionable.
"""

_AUTO_EVAL_SYSTEM = """\
Evaluate whether the following coding task has been fully completed based on the
agent's final response. Output ONLY one of:
- "COMPLETE"   — the task is fully done with no remaining steps.
- "INCOMPLETE" — the task is not done or only partially done.
"""


def run_auto(prompt: str, agent) -> str:
    """
    @auto modifier — autonomous orchestrator that drives the existing agent to
    completion without any user interaction.

    Behaviour:
    - Streams LLM output, tool calls and results live (same as normal mode).
    - Auto-accepts all tool confirmations (same as @insecure).
    - Intercepts ask_user calls, shows the question + options, and answers via LLM.
    - Evaluates task completion after each agent run and continues if needed.
    """
    import threading
    import display
    from agent import Agent, get_activity_hint
    import agent as agent_module
    from api_client import chat_nonstream

    MAX_AUTO_ITERATIONS = 5

    display.print_info("[@auto] Autonomous mode — no user input required.")

    history: list[str] = []

    # ── Orchestrator: answers ask_user questions via LLM ─────────────────────
    def _orchestrate(question: str, options: list) -> str:
        display.stream_stop()
        display.spinner_stop()

        display.print_info(f"[@auto] Agent is asking: {question}")
        for i, o in enumerate(options):
            display.print_info(f"  {i + 1}. {o}")
        display.print_info("[@auto] Orchestrator deciding…")

        opts_lines = "\n".join(f"  {i + 1}. {o}" for i, o in enumerate(options)) if options else ""
        history_str = "\n".join(history[-5:]) if history else "None"

        full_prompt = (
            f"{_AUTO_ORCHESTRATOR_SYSTEM}\n\n"
            f"Original task: {prompt}\n\n"
            f"Recent progress:\n{history_str}\n\n"
            f"The agent is asking: {question}\n"
            + (f"Options:\n{opts_lines}\n\n" if opts_lines else "\n")
            + "Your answer:"
        )
        try:
            result = chat_nonstream(
                full_prompt,
                conversation_id=agent.conversation_id + "-auto-orch",
                agent_id=agent.selected_agent_id,
                selected_model=agent.selected_model,
            )
            answer = result.get("message", "").strip()
        except Exception:
            answer = options[0] if options else "yes"

        display.print_info(f"[@auto] Orchestrator chose: {answer!r}")
        history.append(f"Agent asked: {question!r} → answered: {answer!r}")
        return answer

    # ── Completion evaluator ──────────────────────────────────────────────────
    def _is_complete(task: str, response: str) -> bool:
        full_prompt = (
            f"{_AUTO_EVAL_SYSTEM}\n\n"
            f"Task: {task}\n\n"
            f"Agent's final response:\n{response}"
        )
        try:
            result = chat_nonstream(
                full_prompt,
                conversation_id=agent.conversation_id + "-auto-eval",
                agent_id=agent.selected_agent_id,
                selected_model=agent.selected_model,
            )
            verdict = result.get("message", "").strip().upper()
            return verdict.startswith("COMPLETE")
        except Exception:
            return True  # assume complete on error

    # ── Patch ask_user in the global tool registry ────────────────────────────
    original_ask_user = agent_module.TOOL_REGISTRY.get("ask_user")
    agent_module.TOOL_REGISTRY["ask_user"] = _orchestrate

    try:
        # ── Display callbacks (same wiring as repl._process) ─────────────────
        def _on_llm_start(in_chars: int = 0):
            display.stream_start(in_chars=in_chars)

        def _on_llm_chunk(text: str):
            display.stream_chunk(text)

        def _on_token_count(in_t: int, out_t: int):
            display.stream_tokens(in_t, out_t)

        def _on_thinking(text: str):
            display.stream_stop()
            display.print_thinking(text)
            display.stream_start()

        def _on_tool_use(name: str, params: dict):
            display.stream_stop()
            display.print_tool_use(name, params)
            if name != "ask_user":
                display.spinner_start(f"Running {name}…", status=name)

        def _on_tool_result(name: str, result: str):
            if name != "ask_user":
                display.spinner_stop()
            display.print_tool_result(name, result)

        def _on_bash_output(line: str, is_stderr: bool = False):
            display.bash_output(line, is_stderr)

        def _on_llm_activity(content: str):
            def _fetch():
                hint = get_activity_hint(content)
                if hint and display._stream_view is not None:
                    display._stream_view.hint = hint
                    if not display._stream_view.activities or display._stream_view.activities[-1] != hint:
                        display._stream_view.activities.append(hint)
            threading.Thread(target=_fetch, daemon=True).start()

        # Single persistent worker with insecure mode (auto-accept all tools)
        worker = Agent(on_confirm_tool=lambda n, p: True)
        worker.selected_model      = agent.selected_model
        worker.selected_model_name = agent.selected_model_name
        worker.selected_agent_id   = agent.selected_agent_id
        worker.selected_agent_name = agent.selected_agent_name
        worker.on_llm_start    = _on_llm_start
        worker.on_llm_chunk    = _on_llm_chunk
        worker.on_token_count  = _on_token_count
        worker.on_thinking     = _on_thinking
        worker.on_tool_use     = _on_tool_use
        worker.on_tool_result  = _on_tool_result
        worker.on_bash_output  = _on_bash_output
        worker.on_llm_activity = _on_llm_activity

        last_response = ""
        current_prompt = prompt

        for iteration in range(MAX_AUTO_ITERATIONS):
            display.print_info(f"[@auto] Iteration {iteration + 1}/{MAX_AUTO_ITERATIONS}…")
            if iteration > 0:
                display.print_info(f"[@auto] Sending to agent: {current_prompt[:200]}…")

            try:
                response = worker.run(current_prompt)
            except Exception as e:
                display.stream_stop()
                display.print_info(f"[@auto] Agent error: {e}")
                break

            display.stream_stop()
            last_response = response
            history.append(f"Iteration {iteration + 1}: {response[:300]}")

            # Show intermediate responses so the user can follow progress
            display.print_separator()
            display.print_response(response)

            if _is_complete(prompt, response):
                display.print_info("[@auto] Task completed successfully.")
                # Return empty string — response was already printed above
                return ""

            display.print_info("[@auto] Task not yet complete — continuing…")
            current_prompt = (
                f"[Original task: {prompt}]\n\n"
                f"Your last response was:\n{response}\n\n"
                f"The task is not yet fully complete. "
                f"Continue from where you left off and finish it."
            )

        display.print_info("[@auto] Max iterations reached.")
        # Last response already printed in the loop
        return ""

    finally:
        display.stream_stop()
        display.spinner_stop()
        if original_ask_user is not None:
            agent_module.TOOL_REGISTRY["ask_user"] = original_ask_user
        elif "ask_user" in agent_module.TOOL_REGISTRY:
            del agent_module.TOOL_REGISTRY["ask_user"]


# ---------------------------------------------------------------------------
# @reason modifier
# ---------------------------------------------------------------------------


def run_reason(prompt: str, agent, use_llm_for_ask_user: bool = False) -> str:
    '''
    @reason modifier — like @auto but WITHOUT auto-accepting tool confirmations.

    Behaviour:
    - Streams LLM output, tool calls and results live (same as normal mode).
    - Keeps normal tool confirmation prompts for mutating tools.
    - Intercepts ask_user calls:
      - When use_llm_for_ask_user=True (explicit @reason modifier), shows the
        question + options and answers via LLM.
      - When use_llm_for_ask_user=False (normal REPL flow), routes the question
        to the CLI user via the original ask_user tool.
    - Evaluates task completion after each agent run and continues if needed.
    - Ensures the agent always provides a final answer to the user's question.
    '''
    import threading
    import display
    from agent import Agent, get_activity_hint
    import agent as agent_module
    from api_client import chat_nonstream

    MAX_REASON_ITERATIONS = agent.MAX_TOOL_ITERATIONS

    display.print_info('[@reason] Reasoning mode — tool confirmations still required.')

    history: list[str] = []

    # ── Orchestrator: handle ask_user questions ─────────────
    def _orchestrate(question: str, options: list) -> str:
        display.stream_stop()
        display.spinner_stop()

        display.print_info(f'[@reason] Agent is asking: {question}')
        for i, o in enumerate(options):
            display.print_info(f'  {i + 1}. {o}')

        if use_llm_for_ask_user:
            display.print_info('[@auto] Orchestrator deciding…')

            opts_lines = '\n'.join(f'  {i + 1}. {o}' for i, o in enumerate(options)) if options else ''
            history_str = '\n'.join(history[-5:]) if history else 'None'

            full_prompt = (
                f'{_AUTO_ORCHESTRATOR_SYSTEM}\n\n'
                f'Original task: {prompt}\n\n'
                f'Recent progress:\n{history_str}\n\n'
                f'The agent is asking: {question}\n'
                + (f'Options:\n{opts_lines}\n\n' if opts_lines else '\n')
                + 'Your answer:'
            )
            try:
                result = chat_nonstream(
                    full_prompt,
                    conversation_id=agent.conversation_id + '-reason-orch',
                    agent_id=agent.selected_agent_id,
                    selected_model=agent.selected_model,
                )
                answer = result.get('message', '').strip()
            except Exception:
                answer = options[0] if options else 'yes'

            display.print_info(f'[@auto] Orchestrator chose: {answer!r}')
        else:
            if original_ask_user is not None:
                answer = original_ask_user(question, options)
            else:
                # Fallback: choose first option or 'yes' if none provided
                answer = options[0] if options else 'yes'

            display.print_info(f'[@reason] User chose: {answer!r}')

        history.append(f'Agent asked: {question!r} → answered: {answer!r}')
        return answer

    # ── Completion evaluator ──────────────────────────────────────────────────
    def _is_complete(task: str, response: str) -> bool:
        full_prompt = (
            f"{_AUTO_EVAL_SYSTEM}\n\n"
            f"Task: {task}\n\n"
            f"Agent's final response:\n{response}"
        )
        try:
            result = chat_nonstream(
                full_prompt,
                conversation_id=agent.conversation_id + "-reason-eval",
                agent_id=agent.selected_agent_id,
                selected_model=agent.selected_model,
            )
            verdict = result.get("message", "").strip().upper()
            return verdict.startswith("COMPLETE")
        except Exception:
            return True  # assume complete on error

    # ── Patch ask_user in the global tool registry ────────────────────────────
    original_ask_user = agent_module.TOOL_REGISTRY.get("ask_user")
    agent_module.TOOL_REGISTRY["ask_user"] = _orchestrate

    try:
        # ── Display callbacks (same wiring as repl._process) ─────────────────
        def _on_llm_start(in_chars: int = 0):
            display.stream_start(in_chars=in_chars)

        def _on_llm_chunk(text: str):
            display.stream_chunk(text)

        def _on_token_count(in_t: int, out_t: int):
            display.stream_tokens(in_t, out_t)

        def _on_thinking(text: str):
            display.stream_stop()
            display.print_thinking(text)
            display.stream_start()

        def _on_tool_use(name: str, params: dict):
            display.stream_stop()
            display.print_tool_use(name, params)
            if name != "ask_user":
                display.spinner_start(f"Running {name}…", status=name)

        def _on_tool_result(name: str, result: str):
            if name != "ask_user":
                display.spinner_stop()
            display.print_tool_result(name, result)

        def _on_bash_output(line: str, is_stderr: bool = False):
            display.bash_output(line, is_stderr)

        def _on_llm_activity(content: str):
            def _fetch():
                hint = get_activity_hint(content)
                if hint and display._stream_view is not None:
                    display._stream_view.hint = hint
                    if not display._stream_view.activities or display._stream_view.activities[-1] != hint:
                        display._stream_view.activities.append(hint)
            threading.Thread(target=_fetch, daemon=True).start()

        # Worker preserves the ORIGINAL on_confirm_tool from the caller agent
        worker = Agent(on_confirm_tool=agent.on_confirm_tool)
        worker.selected_model      = agent.selected_model
        worker.selected_model_name = agent.selected_model_name
        worker.selected_agent_id   = agent.selected_agent_id
        worker.selected_agent_name = agent.selected_agent_name
        worker.on_llm_start    = _on_llm_start
        worker.on_llm_chunk    = _on_llm_chunk
        worker.on_token_count  = _on_token_count
        worker.on_thinking     = _on_thinking
        worker.on_tool_use     = _on_tool_use
        worker.on_tool_result  = _on_tool_result
        worker.on_bash_output  = _on_bash_output
        worker.on_llm_activity = _on_llm_activity

        last_response = ""
        current_prompt = prompt

        for iteration in range(MAX_REASON_ITERATIONS):
            display.print_info(f"[@reason] Iteration {iteration + 1}/{MAX_REASON_ITERATIONS}…")
            if iteration > 0:
                display.print_info(f"[@reason] Sending to agent: {current_prompt[:200]}…")

            try:
                response = worker.run(current_prompt)
            except Exception as e:
                display.stream_stop()
                display.print_info(f"[@reason] Agent error: {e}")
                break

            display.stream_stop()
            last_response = response
            history.append(f"Iteration {iteration + 1}: {response[:300]}")

            display.print_separator()
            display.print_response(response)

            if _is_complete(prompt, response):
                display.print_info("[@reason] Task completed.")
                return ""

            display.print_info("[@reason] Task not yet complete — continuing…")
            current_prompt = (
                f"[Original task: {prompt}]\n\n"
                f"Your last response was:\n{response}\n\n"
                f"The task is not yet fully complete. "
                f"Continue from where you left off and finish it."
            )

        display.print_info("[@reason] Max iterations reached.")
        return ""

    finally:
        display.stream_stop()
        display.spinner_stop()
        if original_ask_user is not None:
            agent_module.TOOL_REGISTRY["ask_user"] = original_ask_user
        elif "ask_user" in agent_module.TOOL_REGISTRY:
            del agent_module.TOOL_REGISTRY["ask_user"]


# ---------------------------------------------------------------------------
# Modifier dispatch table — add new @modifiers here
# ---------------------------------------------------------------------------

MODIFIERS: dict[str, callable] = {
    "multi":  run_multi,
    "auto":   run_auto,
    "reason": lambda prompt, agent: run_reason(prompt, agent, use_llm_for_ask_user=True),
}

# ---------------------------------------------------------------------------
# Passthrough modifiers — handled in repl._process, not via MODIFIERS dispatch
# ---------------------------------------------------------------------------
# @insecure  — auto-accepts all tool confirmations (write_file, edit_file, bash)
#              without prompting the user. Use with caution.

PASSTHROUGH_MODIFIERS: set[str] = {"insecure"}


def apply_modifiers(modifiers: list[str], prompt: str, agent) -> Optional[str]:
    """
    Apply the first recognised modifier to *prompt* and return the result.
    Returns None if no known modifier is in the list (caller should use normal flow).
    """
    for name in modifiers:
        handler = MODIFIERS.get(name)
        if handler:
            return handler(prompt, agent)
    return None
