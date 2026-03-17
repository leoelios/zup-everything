"""
Modifier system — @modifier prefix handling for user prompts.

Modifiers start with @ and can appear anywhere in the prompt.
Example: "@multi analyze all Python files in this project"
"""

import json
import re
import concurrent.futures
from typing import Optional

# Matches @word tokens anywhere in the prompt
MODIFIER_RE = re.compile(r'@(\w+)\b', re.IGNORECASE)


def extract_modifiers(prompt: str) -> tuple[list[str], str]:
    """
    Extract @modifier tokens from a prompt.
    Returns (list_of_modifier_names_lowercased, prompt_with_modifiers_removed).
    """
    found: list[str] = []

    def _collect(m: re.Match) -> str:
        found.append(m.group(1).lower())
        return ''

    clean = MODIFIER_RE.sub(_collect, prompt).strip()
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
# Modifier dispatch table — add new @modifiers here
# ---------------------------------------------------------------------------

MODIFIERS: dict[str, callable] = {
    "multi": run_multi,
}


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
