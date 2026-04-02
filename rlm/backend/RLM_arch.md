# RLM Architecture and End-to-End Flow

This document explains how the RLM system works in this repository, with a full flow graph and a concrete example run.

## 1) What RLM Means Here

RLM is an iterative Root-LLM + REPL + Sub-LLM loop:

- Root LLM decides the next action.
- It writes REPL code blocks.
- REPL executes code against external context (not blindly injected each turn).
- REPL can call Sub-LLM through llm_query(...).
- REPL outputs are fed back into the Root LLM message stack.
- The loop continues until Root LLM returns FINAL(...) or FINAL_VAR(...).

## 2) Complete System Graph

```mermaid
flowchart TD
    A[User in Streamlit Query Runner] --> B[POST /api/v1/rlm/query]
    B --> C[Route: routes/query_route.py]
    C --> D[RLM_REPL.completion(context, query)]

    D --> E[Setup phase]
    E --> E1[Build system prompt]
    E --> E2[Store context in REPL variable: context]
    E --> E3[Inject llm_query function into REPL]

    D --> F[Iteration loop]
    F --> G[Build next_action_prompt(query, iteration)]
    G --> H[Root LLM call with message stack]
    H --> I[Root LLM response]

    I --> J{Contains ```repl``` code?}
    J -- No --> K[Append assistant response to messages]
    J -- Yes --> L[Execute each REPL code block]

    L --> M{Code calls llm_query?}
    M -- No --> N[Collect REPL stdout/stderr]
    M -- Yes --> O[Sub-LLM call]
    O --> P[Sub-LLM response]
    P --> N

    N --> Q[Append REPL feedback into messages]
    K --> R[Check FINAL(...) or FINAL_VAR(...)]
    Q --> R

    R -- Not final --> F
    R -- Final --> S[Return final answer]

    S --> T[Route builds debug payload + trace rows]
    T --> U[API response: success/answer/cost/debug]
    U --> V[Streamlit renders Final Answer + Step-by-Step Trace]

    D --> W[Tracer JSONL logs per turn]
    W --> T
```

## 3) Data Movement: What Is Sent to Root LLM

At each Root LLM turn, the payload contains:

- Existing message stack (system + prior feedback messages)
- New user_prompt from next_action_prompt(query, iteration)

Important nuance:

- Full raw context is not re-sent as a giant direct prompt every turn.
- Context lives in REPL as context.
- If Root asks REPL to print/read/summarize context, that REPL output is appended to messages.
- Therefore, context and Sub-LLM outputs can still reach Root LLM indirectly through REPL feedback messages.

## 4) Data Movement: What Is Sent to Sub-LLM

Sub-LLM receives only what REPL passes to llm_query(...), usually a focused prompt like:

- Summarize this chunk
- Explain this extracted symbol
- Generate a focused answer for one sub-problem

Sub-LLM response is returned to REPL, then can become part of REPL output and later part of Root LLM input messages.

## 5) Step-by-Step Example Use Case

Use case:

- Query: How many lines are in this code and what does it do?
- Context: A Python class file with multiple methods.

Expected loop:

1. Iteration 0 (Root LLM)
- Root emits REPL code to inspect context, maybe print part of context.
- REPL executes and returns output.
- Output appended to messages.

2. Iteration 1 (Root LLM)
- Root now sees REPL output and writes analysis code.
- May call llm_query(...) for deeper semantic summary.
- Sub-LLM returns detailed summary.
- REPL stores result in variable (example: answer).
- REPL output and variables appended to messages.

3. Iteration 2 (Root LLM)
- Root sees prior REPL + Sub-LLM-derived feedback.
- Produces refined final answer.
- Emits FINAL(...) or references FINAL_VAR(answer).

4. Return
- API returns answer, cost summary, and debug trace rows.
- Frontend shows:
  - Final Answer
  - Root response per step
  - REPL code blocks
  - REPL execution results
  - REPL variable snapshot
  - Sub-call provenance (Iteration/Turn/Sub-call)
  - Exact root input payload per step

## 6) Component Mapping in This Repository

- Root loop and orchestration: core/engine.py
- REPL runtime and llm_query hook: core/repl_env.py
- API query endpoint: routes/query_route.py
- Per-turn tracing: utils/tracing.py
- Frontend debugger view: frontend/app.py
- Server entrypoint: main.py

## 7) Why This Design Is Useful

- Handles long context without stuffing everything into one prompt.
- Enables iterative tool-augmented reasoning.
- Allows focused Sub-LLM delegation for expensive analysis.
- Provides rich observability (trace rows, payload visibility, sub-call provenance).

## 8) Mental Model

Think of the Root LLM as a planner and verifier, REPL as the execution workspace, and Sub-LLM as a specialist helper. The final decision boundary remains at the Root loop when it emits FINAL(...) or FINAL_VAR(...).

## 9) Complete Graph Example: 100-Line Code Context

Example scenario:

- Context: A 100-line Python file (classes + helper functions)
- User query: How many lines are there and what does this file do?

```mermaid
flowchart TD
    A[User query + 100-line code] --> B[/api/v1/rlm/query]
    B --> C[Create RLM_REPL]
    C --> D[REPL context = full 100-line code]
    D --> E[Iteration 0: Root LLM]

    E --> E1[Root emits REPL code: inspect context]
    E1 --> E2[REPL executes: line count + quick sample]
    E2 --> E3[Append REPL output to root messages]

    E3 --> F[Iteration 1: Root LLM]
    F --> F1[Root emits REPL code: extract symbols/functions]
    F1 --> F2[REPL executes and calls llm_query on extracted summary]
    F2 --> F3[Sub-LLM returns semantic summary]
    F3 --> F4[REPL stores summary in variables]
    F4 --> F5[Append REPL output (incl. sub summary) to root messages]

    F5 --> G[Iteration 2: Root LLM]
    G --> G1[Root combines line count + semantic summary]
    G1 --> G2[Root emits FINAL(...) or FINAL_VAR(answer)]

    G2 --> H[Return final answer]
    H --> I[Frontend Step-by-Step Trace]
    I --> I1[Shows root input payload per iteration]
    I --> I2[Shows sub-call provenance Iteration/Turn/Sub-call]
    I --> I3[Shows REPL results and variables]
```

## 10) Concrete Iteration Trace (Same 100-Line Example)

1. Iteration 0
- Root input includes: system message + user prompt for iteration 0.
- Root output code (example):

```repl
lines = context.splitlines()
line_count = len(lines)
sample = "\n".join(lines[:12])
print(line_count)
print(sample)
```

- REPL output sent back to Root on next turn:
  - `100`
  - first few lines of code sample

2. Iteration 1
- Root input now includes prior REPL output message.
- Root output code (example):

```repl
functions = [ln for ln in context.splitlines() if ln.strip().startswith("def ")]
classes = [ln for ln in context.splitlines() if ln.strip().startswith("class ")]
summary_prompt = f"Classes: {classes}\nFunctions: {functions[:20]}\nExplain behavior briefly."
semantic = llm_query(summary_prompt)
print(semantic)
```

- Sub-LLM is called once through `llm_query`.
- Sub-LLM response is printed by REPL and then appended into Root message stack.

3. Iteration 2
- Root input now includes:
  - line count evidence from Iteration 0
  - semantic summary evidence from Iteration 1 (via REPL output)
- Root emits final result:

```text
FINAL("The file has 100 lines and implements ...")
```

Key takeaway:

- The whole 100-line context is available in REPL as `context`.
- Root LLM gets what is in the message stack each turn.
- Sub-LLM output reaches Root only after it is produced in REPL and appended as feedback.
