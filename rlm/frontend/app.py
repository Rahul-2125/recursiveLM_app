import json
import html
import ast
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ========== CONSTANTS ==========
APP_TITLE = "ContextPilot"
DEFAULT_API_BASE = "http://localhost:8000/api/v1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"


# ========== PAGE CONFIGURATION ==========
def _configure_page():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _apply_styling():
    bg = "#0b1220"
    card = "#111a2c"
    surface = "#1b263b"
    text = "#e5e7eb"
    muted = "#9ca3af"
    accent = "#14b8a6"
    border = "#2f3f5a"
    sidebar_bg = (
        "linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(11, 18, 32, 0.95))"
    )

    css = """
<style>
    :root {
        --bg: __BG__;
        --card: __CARD__;
        --surface: __SURFACE__;
        --text: __TEXT__;
        --muted: __MUTED__;
        --accent: __ACCENT__;
        --border: __BORDER__;
    }

    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, rgba(20, 184, 166, 0.12), transparent 28%), var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: __SIDEBAR_BG__;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    [data-testid="stAppViewContainer"] section[data-testid="stSidebar"] [data-baseweb="input"] > div,
    [data-testid="stAppViewContainer"] section[data-testid="stSidebar"] [data-baseweb="textarea"] > div,
    [data-testid="stAppViewContainer"] section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: var(--surface) !important;
        border-color: var(--border) !important;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
    }

    .cm-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }

    .cm-kpi {
        display: flex;
        gap: 10px;
        align-items: baseline;
    }

    .cm-kpi-label {
        color: var(--muted);
        font-size: 0.85rem;
    }

    .cm-kpi-value {
        color: var(--text);
        font-size: 1.35rem;
        font-weight: 600;
    }

    .cm-section-title {
        font-size: 1.06rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: var(--text);
    }

    .cm-muted {
        color: var(--muted);
        font-size: 0.9rem;
    }

    .tab-hero {
        border-radius: 14px;
        padding: 12px 14px;
        margin: 0.2rem 0 0.8rem 0;
        border: 1px solid var(--border);
        background: linear-gradient(120deg, rgba(27, 38, 59, 0.95), rgba(11, 18, 32, 0.95));
    }

    .tab-hero-title {
        font-size: 1.02rem;
        font-weight: 700;
        line-height: 1.3;
    }

    .tab-hero-sub {
        margin-top: 0.2rem;
        color: var(--muted);
        font-size: 0.88rem;
    }

    .tab-hero.query {
        border-left: 4px solid #14b8a6;
        box-shadow: 0 10px 20px rgba(20, 184, 166, 0.12);
    }

    .tab-hero.context {
        border-left: 4px solid #60a5fa;
        box-shadow: 0 10px 20px rgba(96, 165, 250, 0.12);
    }

    .tab-hero.system {
        border-left: 4px solid #f59e0b;
        box-shadow: 0 10px 20px rgba(245, 158, 11, 0.12);
    }

    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: var(--text);
    }

    [data-baseweb="input"] > div,
    [data-baseweb="select"] > div,
    [data-baseweb="textarea"] > div,
    textarea,
    input {
        background-color: var(--surface) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }

    textarea::placeholder,
    input::placeholder {
        color: var(--muted) !important;
        opacity: 0.9 !important;
    }

    [data-baseweb="select"] svg {
        fill: var(--text) !important;
    }

    [data-baseweb="tab-list"] {
        gap: 0.35rem;
    }

    button[kind],
    .stButton > button {
        border-radius: 999px;
        border: 1px solid var(--border) !important;
        background-color: var(--accent) !important;
        color: #ffffff !important;
        box-shadow: 0 8px 18px rgba(15, 118, 110, 0.35);
    }

    button[kind]:hover,
    .stButton > button:hover {
        filter: brightness(1.03);
        box-shadow: 0 12px 26px rgba(15, 118, 110, 0.45);
    }

    [data-testid="stNumberInput"] button {
        background-color: var(--surface) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
        box-shadow: none !important;
        border-radius: 0 !important;
    }

    [data-testid="stNumberInput"] button:hover {
        background-color: #263757 !important;
        color: var(--text) !important;
    }

    [data-baseweb="tab"] {
        color: var(--muted) !important;
        border-bottom: 2px solid transparent !important;
        position: relative;
    }

    [data-baseweb="tab"]:not(:last-child) {
        margin-right: 1.1rem;
    }

    [data-baseweb="tab"]:not(:last-child)::after {
        content: "|";
        position: absolute;
        right: -0.72rem;
        top: 50%;
        transform: translateY(-50%);
        color: var(--muted);
        opacity: 0.7;
        pointer-events: none;
        font-weight: 500;
    }

    [aria-selected="true"][data-baseweb="tab"] {
        color: var(--accent) !important;
        border-bottom-color: var(--accent) !important;
    }

    [data-testid="stTabs"] [role="tabpanel"]:nth-of-type(1) {
        border-top: 1px solid rgba(20, 184, 166, 0.28);
    }

    [data-testid="stTabs"] [role="tabpanel"]:nth-of-type(2) {
        border-top: 1px solid rgba(96, 165, 250, 0.28);
    }

    [data-testid="stTabs"] [role="tabpanel"]:nth-of-type(3) {
        border-top: 1px solid rgba(245, 158, 11, 0.28);
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        background-color: var(--card) !important;
    }

    [data-testid="stExpander"] summary {
        background-color: var(--surface) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
    }

    [data-testid="stExpander"] summary svg {
        fill: var(--text) !important;
    }

    [data-testid="stCodeBlock"] {
        background-color: var(--surface) !important;
        border-radius: 10px;
        border: 1px solid var(--border);
    }

    [data-testid="stCode"],
    .stCode {
        background-color: var(--surface) !important;
        border-radius: 10px;
        border: 1px solid var(--border);
    }

    [data-testid="stCodeBlock"] pre,
    [data-testid="stCodeBlock"] code,
    [data-testid="stCode"] pre,
    [data-testid="stCode"] code,
    .stCode pre,
    .stCode code,
    .stCode div,
    .stCode span,
    pre code,
    .hljs {
        background-color: transparent !important;
        color: var(--text) !important;
    }

    [data-testid="stCode"] button,
    .stCode button {
        background-color: var(--accent) !important;
        color: #ffffff !important;
        border: 1px solid var(--border) !important;
        box-shadow: none !important;
        border-radius: 999px !important;
    }

    [data-testid="stJson"] {
        background-color: var(--surface) !important;
        border-radius: 10px;
        border: 1px solid var(--border);
        padding: 0.5rem 0.75rem;
    }

    [data-testid="stJson"] pre,
    [data-testid="stJson"] code,
    [data-testid="stJson"] span {
        background-color: transparent !important;
        color: var(--text) !important;
        border-radius: 10px;
    }

    [data-testid="stDataFrame"] {
        background-color: var(--card) !important;
        border-radius: 10px;
        border: 1px solid var(--border);
        padding: 0.25rem;
    }

    [data-testid="stDataFrame"] *,
    [data-testid="stDataFrame"] th,
    [data-testid="stDataFrame"] td {
        color: var(--text) !important;
        background-color: transparent !important;
    }

    [data-testid="stAlert"] {
        border: 1px solid var(--border);
    }
</style>
"""
    css = (
        css.replace("__BG__", bg)
        .replace("__CARD__", card)
        .replace("__SURFACE__", surface)
        .replace("__TEXT__", text)
        .replace("__MUTED__", muted)
        .replace("__ACCENT__", accent)
        .replace("__BORDER__", border)
        .replace("__SIDEBAR_BG__", sidebar_bg)
    )
    st.markdown(css, unsafe_allow_html=True)


def _render_query_debug(debug_payload: Dict[str, Any]):
    """Render full execution details for a single query run."""
    context_info = debug_payload.get("context", {})
    trace_rows = debug_payload.get("trace_rows", [])
    input_query = context_info.get("query", "")
    context_preview = context_info.get("preview", "")
    context_preview_truncated = bool(context_info.get("preview_truncated", False))

    if trace_rows:
        st.markdown("#### Step-by-Step Trace")
        for idx, row in enumerate(trace_rows, start=1):
            iteration = row.get("iteration")
            turn = row.get("turn")
            final_flag = " | FINAL" if row.get("final_answer") else ""
            with st.expander(
                f"Step {idx} | Iteration {iteration} | Turn {turn}{final_flag}"
            ):
                if idx == 1:
                    st.markdown("**Input**")
                    st.markdown("User Query (sent to Root LLM)")
                    st.code(input_query or "", language="text")
                    st.markdown("User Provided Context (stored in REPL)")
                    st.code(context_preview or "", language="text")
                    if context_preview_truncated:
                        st.caption(
                            "Context preview truncated to first 4000 characters."
                        )

                root_input = row.get("root_llm_input", {}) or {}
                if root_input:
                    st.markdown("**Data Sent to Root LLM (this step)**")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Messages Sent", int(root_input.get("message_count", 0)))
                    r2.metric("Total Characters", int(root_input.get("total_chars", 0)))
                    r3.metric(
                        "Contains REPL Feedback",
                        "Yes" if root_input.get("contains_repl_feedback") else "No",
                    )
                    r4.metric(
                        "Contains SubLLM Feedback",
                        "Yes" if root_input.get("contains_sub_llm_feedback") else "No",
                    )
                    # st.caption(
                    #     "Raw message contents below are exact payload entries sent to Root LLM for this step."
                    # )

                    with st.expander("View Root LLM Input Messages"):
                        for msg in root_input.get("messages", []):
                            msg_idx = msg.get("index", "?")
                            role = msg.get("role", "unknown")
                            chars = msg.get("chars", 0)
                            has_repl = msg.get("contains_repl_output", False)
                            has_sub = msg.get("contains_sub_llm_output", False)
                            msg_title = (
                                f"Message {msg_idx} | {role.title()} | "
                                f"{chars} chars"
                            )
                            with st.expander(msg_title):
                                c1, c2, c3 = st.columns(3)
                                c1.metric(
                                    "Contains REPL Output", "Yes" if has_repl else "No"
                                )
                                c2.metric(
                                    "Contains SubLLM Output", "Yes" if has_sub else "No"
                                )
                                c3.metric("Role", role.title())

                                st.markdown("Message Content")
                                safe_content = html.escape(
                                    str(msg.get("content", ""))
                                ).replace("\n", "<br>")
                                st.markdown(
                                    f"""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px;max-height:260px;overflow:auto;color:var(--text);font-family:Consolas,'Courier New',monospace;white-space:pre-wrap;line-height:1.45;">
  {safe_content}
</div>
""",
                                    unsafe_allow_html=True,
                                )

                st.markdown("**Root LLM Response**")
                st.code(row.get("response", ""), language="text")

                code_blocks = row.get("code_blocks", [])
                if code_blocks:
                    st.markdown("**REPL Code Blocks**")
                    for i, code in enumerate(code_blocks, start=1):
                        st.code(code, language="python")
                else:
                    st.caption("No REPL code blocks in this step.")

                execution_results = row.get("execution_results", [])
                sub_llm_calls = row.get("sub_llm_calls", [])
                repl_state = row.get("repl_state", {}) or {}
                repl_vars = repl_state.get("local_vars", [])
                repl_var_values = repl_state.get("local_var_values", {}) or {}

                def _decode_repl_value(value: Any) -> str:
                    """Decode repr-style values for user-friendly rendering."""
                    text = str(value)
                    if len(text) >= 2 and text[0] in ("'", '"') and text[-1] == text[0]:
                        try:
                            parsed = ast.literal_eval(text)
                            if isinstance(parsed, str):
                                return parsed
                        except Exception:
                            return text[1:-1]
                    return text

                raw_context_value = _decode_repl_value(
                    repl_var_values.get("context", "")
                ).strip()
                canonical_context_value = (context_preview or raw_context_value).strip()

                filtered_execution_results = []
                sub_llm_responses = {
                    str(call.get("response", "")).strip()
                    for call in sub_llm_calls
                    if call.get("response")
                }
                for result in execution_results:
                    result_text = str(result).strip()
                    if canonical_context_value and (
                        result_text == canonical_context_value
                        or result_text in canonical_context_value
                        or canonical_context_value in result_text
                    ):
                        continue
                    if result_text in sub_llm_responses:
                        continue
                    filtered_execution_results.append(result)

                if filtered_execution_results or repl_var_values:
                    st.markdown("**REPL Execution Results**")
                    for i, result in enumerate(filtered_execution_results, start=1):
                        st.code(result, language="text")

                    for call in sub_llm_calls:
                        call_idx = call.get("call_index", "?")
                        st.markdown(
                            f"<div style='display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid #0f766e; color:#0f766e; font-weight:700; margin:6px 0;'>Iteration {iteration} | Turn {turn} | Sub-call #{call_idx}</div>",
                            unsafe_allow_html=True,
                        )
                        st.code(
                            f"{call.get('response', '')}",
                            language="text",
                        )

                    if repl_var_values:
                        lines = []
                        for name, value in repl_var_values.items():
                            if name == "context":
                                lines.append(f"context = {canonical_context_value}")
                            else:
                                lines.append(f"{name} = {_decode_repl_value(value)}")
                        st.code("\n".join(lines), language="python")

                st.markdown("**REPL Environment Variables**")
                if repl_vars:
                    st.code("\n".join([str(var) for var in repl_vars]), language="text")
                else:
                    st.caption("No variables found in REPL environment for this step.")

                if sub_llm_calls:
                    st.markdown("**SubLLM called by RootLLM**")
                    st.markdown("**Sub-Agent (Sub-LLM) Calls**")
                    for call in sub_llm_calls:
                        call_title = (
                            f"Iteration {iteration} | Turn {turn} | Sub-call #{call.get('call_index')} "
                            f"| tokens={call.get('tokens', 0)} "
                            f"| cost=${call.get('cost', 0.0):.6f}"
                        )
                        with st.expander(call_title):
                            st.markdown("SubLLM Input")
                            st.code(call.get("prompt", ""), language="text")
                            st.markdown("SubLLM Response")
                            st.code(call.get("response", ""), language="text")

                if row.get("final_answer"):
                    st.markdown("**Final Answer Found In This Step**")
                    st.code(str(row.get("final_answer")), language="text")
    else:
        st.info("No trace steps available for this run.")


def _normalize_final_answer(answer: Any) -> str:
    """Normalize final answer text for user-friendly display."""
    text = "" if answer is None else str(answer).strip()

    # Handle occasional wrappers such as FINAL(...) that may leak through.
    if text.startswith("FINAL(") and text.endswith(")"):
        text = text[6:-1].strip()

    # Remove one layer of wrapping quotes if present.
    if len(text) >= 2 and text[0] in ("'", '"') and text[-1] == text[0]:
        text = text[1:-1]

    return text.strip()


def _render_final_answer(answer: Any):
    """Render final answer in a clean, readable card."""
    final_text = _normalize_final_answer(answer)
    if not final_text:
        st.info("No final answer returned.")
        return

    safe_text = html.escape(final_text).replace("\n", "<br>")
    st.markdown("### Final Response")
    st.markdown(
        f"""
<div class="cm-card" style="border-left: 4px solid #0f766e; margin-top: 0.25rem;">
    <div style="font-size: 1.02rem; line-height: 1.65; color: var(--text); white-space: normal;">{safe_text}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------- Helpers ----------
def api_get(base_url: str, path: str, timeout: int = 90) -> Tuple[bool, Any]:
    try:
        resp = requests.get(f"{base_url}{path}", timeout=timeout)
        if resp.ok:
            return True, resp.json()
        return False, f"{resp.status_code}: {resp.text}"
    except Exception as exc:
        return False, str(exc)


def api_post(
    base_url: str,
    path: str,
    json_payload: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 600,
) -> Tuple[bool, Any]:
    try:
        resp = requests.post(
            f"{base_url}{path}", json=json_payload, params=params, timeout=timeout
        )
        if resp.ok:
            return True, resp.json()
        return False, f"{resp.status_code}: {resp.text}"
    except Exception as exc:
        return False, str(exc)


def iter_sse_events(
    base_url: str,
    path: str,
    json_payload: Optional[Dict[str, Any]] = None,
    timeout: int = 1800,
):
    """Yield parsed SSE events from backend as dictionaries."""
    resp = requests.post(
        f"{base_url}{path}", json=json_payload, timeout=timeout, stream=True
    )
    if not resp.ok:
        raise RuntimeError(f"{resp.status_code}: {resp.text}")

    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = str(raw_line).strip()
        if not line.startswith("data:"):
            continue

        payload = line[5:].strip()
        if not payload:
            continue

        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def parse_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def list_log_files() -> List[Path]:
    if not LOGS_DIR.exists():
        return []
    return sorted(LOGS_DIR.glob("rlm_trace_*.jsonl"), reverse=True)


@st.cache_data(show_spinner=False)
def load_log_dataframe(file_path: str) -> pd.DataFrame:
    rows = parse_jsonl(Path(file_path))
    if not rows:
        return pd.DataFrame()

    normalized: List[Dict[str, Any]] = []
    for row in rows:
        ci = row.get("cost_info") or {}
        normalized.append(
            {
                "timestamp": row.get("timestamp"),
                "session_id": row.get("session_id"),
                "iteration": row.get("iteration"),
                "turn": row.get("turn"),
                "tokens": ci.get("tokens", 0),
                "cost": ci.get("cost", 0.0),
                "input_tokens": ci.get("input_tokens", 0),
                "output_tokens": ci.get("output_tokens", 0),
                "has_final_answer": bool(row.get("final_answer")),
                "code_blocks": len(row.get("code_blocks", [])),
                "execution_results": len(row.get("execution_results", [])),
            }
        )

    df = pd.DataFrame(normalized)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


_configure_page()
_apply_styling()


# ---------- Sidebar ----------
st.sidebar.title(APP_TITLE)
st.sidebar.caption("Recursive Language Models (RLMs) Control Panel")

st.sidebar.divider()

api_base = st.sidebar.text_input("Backend Base URL", value=DEFAULT_API_BASE)


# Cache health check for 30 seconds to reduce API calls
@st.cache_data(ttl=30, show_spinner=False)
def check_health(url: str) -> Tuple[bool, Any]:
    return api_get(url, "/health")


health_ok, health_data = check_health(api_base)

if health_ok:
    st.sidebar.success("Backend Server is Active")
else:
    st.sidebar.error("Backend Server is Unreachable")

st.sidebar.divider()
st.sidebar.markdown("### Quick Actions")
if st.sidebar.button("Refresh Health", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


# ---------- Header ----------
st.title(APP_TITLE)
# st.caption(
#     "User-friendly frontend for querying context, managing files, and analyzing RLM traces"
# )

col1, col2, col3 = st.columns(3)
# with col1:
#     st.markdown(
#         """
# <div class="cm-card">
#   <div class="cm-kpi">
#     <div class="cm-kpi-label">Backend Status</div>
#     <div class="cm-kpi-value">{}</div>
#   </div>
# </div>
# """.format(
#             "Online" if health_ok else "Offline"
#         ),
#         unsafe_allow_html=True,
#     )
# with col2:
#     st.markdown(
#         f"""
# <div class="cm-card">
#   <div class="cm-kpi">
#     <div class="cm-kpi-label">API Base</div>
#     <div class="cm-kpi-value" style="font-size:1rem">{api_base}</div>
#   </div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )
# with col3:
#     st.markdown(
#         f"""
# <div class="cm-card">
#   <div class="cm-kpi">
#     <div class="cm-kpi-label">Time</div>
#     <div class="cm-kpi-value" style="font-size:1rem">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
#   </div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# ---------- Tabs ----------
tab_query, tab_contexts, tab_system = st.tabs(
    ["Query Runner", "Context Manager", "Settings"]
)


with tab_query:
    st.markdown(
        """
<div class="tab-hero query">
  <div class="tab-hero-title">Query Runner Workspace</div>
  <div class="tab-hero-sub">Run query against a selected context and inspect step-by-step RLM traces.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.subheader("Run Query")
    input_mode = st.radio(
        "Context Input", ["Use Context File", "Paste Context Text"], horizontal=True
    )

    query = st.text_area(
        "Query", placeholder="Ask a question about your context...", height=100
    )

    col_q1, col_q2 = st.columns([2, 1])
    with col_q1:
        model = st.text_input(
            "Model (optional)", placeholder="e.g. qwen/qwen2.5-coder-32b-instruct"
        )
    with col_q2:
        max_iterations = st.number_input(
            "Max Iterations", min_value=1, max_value=200, value=30
        )

    stream_steps = st.toggle(
        "Stream step-by-step inference",
        value=True,
        help="Show live turn updates while RLM is running.",
    )

    payload: Dict[str, Any] = {
        "query": query,
        "max_iterations": int(max_iterations),
    }

    if input_mode == "Use Context File":

        @st.cache_data(ttl=20, show_spinner=False)
        def get_context_files_for_selection(url: str):
            return api_get(url, "/rlm/contexts")

        ok_files, files_data = get_context_files_for_selection(api_base)
        files: List[str] = (
            files_data.get("files", [])
            if ok_files and isinstance(files_data, dict)
            else []
        )
        selected_file = st.selectbox(
            "Context File", options=files, index=0 if files else None
        )
        if selected_file:
            payload["context_file"] = selected_file
    else:
        context_text = st.text_area(
            "Context Content",
            height=220,
            placeholder="Paste large context/code here...",
        )
        payload["context"] = context_text

    if model.strip():
        payload["model"] = model.strip()

    live_trace_rendered = False

    if st.button("Run RLM Query", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please provide a query.")
        elif "context" not in payload and "context_file" not in payload:
            st.warning("Please provide context input.")
        else:
            if stream_steps:
                st.info("Streaming live inference...")
                stream_status = st.empty()
                stream_panel = st.empty()
                streamed_rows: List[Dict[str, Any]] = []
                final_data: Optional[Dict[str, Any]] = None
                live_context_preview = payload.get("context", "")
                live_debug_payload: Dict[str, Any] = {
                    "context": {
                        "source": (
                            "inline" if payload.get("context") else "context_file"
                        ),
                        "context_file": payload.get("context_file"),
                        "query": query,
                        "preview": live_context_preview,
                        "chars": (
                            len(live_context_preview)
                            if isinstance(live_context_preview, str)
                            else 0
                        ),
                        "lines": (
                            live_context_preview.count("\n") + 1
                            if isinstance(live_context_preview, str)
                            and live_context_preview
                            else 0
                        ),
                        "preview_truncated": bool(
                            isinstance(live_context_preview, str)
                            and len(live_context_preview) > 4000
                        ),
                    },
                    "trace_rows": [],
                }

                try:
                    for event in iter_sse_events(
                        api_base,
                        "/rlm/query/stream",
                        json_payload=payload,
                        timeout=1800,
                    ):
                        event_type = event.get("type")

                        if event_type == "started":
                            started_context = event.get("context")
                            if isinstance(started_context, dict):
                                live_debug_payload["context"] = started_context
                            stream_status.caption(
                                "Execution started. Waiting for first turn..."
                            )
                            with stream_panel.container():
                                st.markdown("### Live Step-by-Step Trace")
                                _render_query_debug(live_debug_payload)
                            live_trace_rendered = True
                            continue

                        if event_type == "turn":
                            row = event.get("row", {})
                            streamed_rows.append(row)
                            live_debug_payload["trace_rows"] = streamed_rows

                            stream_status.caption(
                                f"Received {len(streamed_rows)} streamed step(s)."
                            )
                            with stream_panel.container():
                                st.markdown("### Live Step-by-Step Trace")
                                _render_query_debug(live_debug_payload)
                            live_trace_rendered = True
                            continue

                        if event_type == "completed":
                            final_data = {
                                "success": bool(event.get("success")),
                                "answer": event.get("answer"),
                                "error": event.get("error"),
                                "cost_summary": event.get("cost_summary"),
                                "debug": event.get("debug"),
                            }
                            break

                        if event_type == "error":
                            st.error(
                                f"Streaming failed: {event.get('message', 'Unknown error')}"
                            )
                            break

                    if final_data:
                        st.session_state["last_query_result"] = final_data
                        if final_data.get("success"):
                            st.success("Query Processed")
                            _render_final_answer(final_data.get("answer", ""))
                        else:
                            st.warning(
                                final_data.get("error", "No final answer returned")
                            )
                    elif streamed_rows:
                        st.warning("Stream ended without final completion payload.")

                except Exception as exc:
                    st.error(f"Streaming request failed: {exc}")
                    with st.spinner("Retrying in non-stream mode..."):
                        ok, data = api_post(
                            api_base, "/rlm/query", json_payload=payload, timeout=1800
                        )

                    if not ok:
                        st.error(f"Fallback request failed: {data}")
                    else:
                        st.session_state["last_query_result"] = data
                        if data.get("success"):
                            st.success("Query Processed")
                            _render_final_answer(data.get("answer", ""))
                        else:
                            st.warning(data.get("error", "No final answer returned"))
            else:
                with st.spinner(
                    "Running query... this may take time for long contexts"
                ):
                    ok, data = api_post(
                        api_base, "/rlm/query", json_payload=payload, timeout=1800
                    )

                if not ok:
                    st.error(f"Request failed: {data}")
                else:
                    st.session_state["last_query_result"] = data
                    if data.get("success"):
                        st.success("Query Processed")
                        _render_final_answer(data.get("answer", ""))
                    else:
                        st.warning(data.get("error", "No final answer returned"))

    last_result = st.session_state.get("last_query_result")
    if (
        isinstance(last_result, dict)
        and last_result.get("debug")
        and not live_trace_rendered
    ):
        _render_query_debug(last_result["debug"])


with tab_contexts:
    st.markdown(
        """
<div class="tab-hero context">
  <div class="tab-hero-title">Context Manager Workspace</div>
  <div class="tab-hero-sub">Browse, upload, and curate context files used by the query runner.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.subheader("Context Files")

    col_c1, col_c2 = st.columns([1, 1])

    with col_c1:
        st.markdown("#### Available Files")

        @st.cache_data(ttl=20, show_spinner=False)
        def get_context_files(url: str):
            return api_get(url, "/rlm/contexts")

        ok, data = get_context_files(api_base)
        if ok:
            files = data.get("files", [])
            st.caption(f"Base Path: {data.get('base_path', '-')}")
            if files:
                file_rows = "".join(
                    [
                        f"<li style='padding:8px 10px; border-bottom:1px solid var(--border); color:var(--text);'>{html.escape(str(file_name))}</li>"
                        for file_name in files
                    ]
                )
                st.markdown(
                    f"""
<div class="cm-card" style="padding:0; overflow:hidden;">
  <div style="padding:10px 12px; color:var(--muted); border-bottom:1px solid var(--border);">file</div>
  <ul style="margin:0; padding:0; list-style:none;">
    {file_rows}
  </ul>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                st.info("No context files found.")
        else:
            st.error(f"Could not fetch files: {data}")

    with col_c2:
        st.markdown("#### Upload / Create Context")
        upload_name = st.text_input(
            "File Name", placeholder="example.py or docs/context.txt"
        )
        upload_content = st.text_area("File Content", height=260)

        if st.button("Save Context File", use_container_width=True):
            if not upload_name.strip():
                st.warning("Please provide file name.")
            else:
                ok_up, data_up = api_post(
                    api_base,
                    "/rlm/contexts/upload",
                    params={"filename": upload_name.strip(), "content": upload_content},
                )
                if ok_up:
                    st.success(f"Saved: {data_up.get('path', upload_name)}")
                    st.rerun()
                else:
                    st.error(f"Upload failed: {data_up}")


with tab_system:
    st.markdown(
        """
<div class="tab-hero system">
  <div class="tab-hero-title">System Monitoring Workspace</div>
  <div class="tab-hero-sub">Track service health and inspect active backend settings.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.subheader("System Status")

    s1, s2 = st.columns(2)

    with s1:
        st.markdown("#### Health Status")

        @st.cache_data(ttl=30, show_spinner=False)
        def get_health_status(url: str):
            return api_get(url, "/health")

        ok_h, data_h = get_health_status(api_base)
        if ok_h:
            st.code(
                json.dumps(data_h["status"].strip('"'), indent=2, ensure_ascii=False),
                language="json",
            )
        else:
            st.error(f"Health check failed: {data_h}")

    with s2:
        st.markdown("#### Backend Settings")

        @st.cache_data(ttl=60, show_spinner=False)
        def get_backend_settings(url: str):
            return api_get(url, "/rlm/settings")

        ok_s, data_s = get_backend_settings(api_base)
        if ok_s:
            st.code(json.dumps(data_s, indent=2, ensure_ascii=False), language="json")
        else:
            st.error(f"Could not load settings: {data_s}")

st.markdown("---")
st.caption(APP_TITLE)
