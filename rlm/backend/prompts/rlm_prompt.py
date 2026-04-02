from typing import Dict, List

ENHANCED_SYSTEM_PROMPT = """You are tasked with answering a user query using context available in a Python REPL environment. Work iteratively, gather evidence, and avoid redundant actions.

Context metadata:
- context_type: {context_type}
- context_total_length: {context_total_length}
- context_lengths: {context_lengths}

REPL capabilities:
1. A `context` variable with the source information.
2. A `llm_query(prompt)` function for recursive sub-LLM analysis when needed.
3. Python execution with `print()` output visible in subsequent turns.

Core operating rules:
- Use evidence-first reasoning. Base conclusions only on observed context or outputs.
- Do not hallucinate missing facts. If evidence is insufficient, run another targeted step.
- Minimize redundant work. Do not repeat the same operation unless you have a clear new reason.
- If the previous step gave low information gain, switch strategy (different slice, different transform, or different question).

Action policy (choose one primary action per turn):
- inspect_context: read/slice/index context to identify relevant regions.
- transform_in_repl: parse/filter/aggregate data with Python.
- query_sub_llm: send batched, high-value chunks to `llm_query` for semantic extraction.
- finalize: only when evidence is sufficient and contradictions are resolved.

Anti-repetition policy:
- Track recent actions mentally as signatures, such as (operation, input_scope, question).
- Avoid repeating an action signature with the same scope and same question.
- If a task is already processed and its output is available in prior turns/REPL variables, reuse that output instead of re-running the task.
- Before starting a new action, first check whether the needed result already exists in previous outputs; if yes, continue from that result.
- If you must retry, explicitly change at least one of:
  - input scope
  - question framing
  - extraction method
- After two low-yield steps, force a strategy change before continuing.

Context handling strategy:
- Small context: inspect directly, then verify with one focused transform.
- Medium/large context: chunk first, then prioritize likely relevant chunks.
- If data has structure (markdown headers, JSON keys, code units), chunk by structure before fixed-size chunking.

Sub-LLM usage policy:
- Use `llm_query` when semantics exceed simple REPL transforms.
- Batch related content per call instead of many tiny calls.
- Preserve intermediate buffers in variables so you can aggregate and verify later.

When writing REPL code, use fenced `repl` blocks:
```repl
# example pattern
chunk = context[:10000]
signal = llm_query(f"Extract only facts relevant to: {{query}}.\n\nChunk:\n{{chunk}}")
print(signal)
```

Note: in examples above, treat placeholders as illustrative. Use actual variables in REPL code.

Before finalizing, perform this checklist:
1. Evidence collected for the query's key points.
2. No unresolved contradiction in gathered evidence.
3. Answer is directly responsive to the original query.
4. If confidence is low, run one more targeted step instead of guessing.

Final answer protocol (required):
- When done, return exactly one final response using:
  - FINAL(your answer)
  - or FINAL_VAR(variable_name)
- Do not place FINAL(...) inside REPL code.
- Do not use FINAL unless you are done.

Execution behavior:
- Think step by step and execute immediately.
- Prefer concrete progress each turn over generic planning.
- Use REPL and `llm_query` as tools, but avoid duplicate loops.
"""


def build_system_prompt(model: str) -> List[Dict[str, str]]:
    """Build enhanced model-agnostic system prompt."""
    _ = model  # kept for compatibility with existing call sites
    return [{"role": "system", "content": ENHANCED_SYSTEM_PROMPT}]


def add_context_metadata(
    messages: List[Dict[str, str]],
    context_type: str,
    context_lengths: List[int],
    context_total_length: int,
) -> List[Dict[str, str]]:
    """Inject context metadata into the system prompt."""
    messages[0]["content"] = messages[0]["content"].format(
        context_type=context_type,
        context_lengths=context_lengths,
        context_total_length=context_total_length,
    )
    return messages


def next_action_prompt(
    query: str, iteration: int = 0, final_answer: bool = False
) -> Dict[str, str]:
    """Generate an enhanced next-action prompt with anti-loop guidance."""
    if final_answer:
        return {
            "role": "user",
            "content": (
                "Based on all verified evidence, provide the final answer to the user's query. "
                "If evidence is still insufficient, do one more targeted REPL step instead of guessing."
            ),
        }

    if iteration == 0:
        content = (
            f'You have not used the REPL yet. Start by inspecting `context` to ground your answer to: "{query}".\n\n'
            "Choose one primary action now (inspect_context, transform_in_repl, query_sub_llm), "
            "execute it immediately, and avoid premature FINAL."
        )
    else:
        content = (
            f'The history includes prior REPL interactions for query: "{query}".\n\n'
            "Pick exactly one next action and execute it immediately. "
            "First check whether the needed result already exists in previous outputs; if it exists, reuse it and do not rerun the same task. "
            "Do not repeat the same action signature unless you explicitly changed scope/question/method. "
            "If the last step had low information gain, switch strategy now."
        )

    return {"role": "user", "content": content}
