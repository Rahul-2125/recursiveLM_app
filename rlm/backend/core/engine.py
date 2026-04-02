"""
Recursive Language Model with REPL environment.
Implements the core RLM algorithm.
"""

from typing import Dict, List, Optional, Any, Union
import re
from core.base import RLM
from core.repl_env import REPLEnv
from utils.tracing import tracer
from services.llm_client import get_llm_client
from prompts.rlm_prompt import (
    build_system_prompt,
    add_context_metadata,
    next_action_prompt,
)
from utils.logging import logger


class RLM_REPL(RLM):
    """
    RLM implementation using REPL environment.
    Context is stored externally in REPL, not passed to model directly.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5",
        recursive_model: str = "gpt-5-mini",
        max_iterations: int = 20,
        max_output_length: int = 500000,
        depth: int = 0,
    ):
        """
        Initialize RLM with REPL.

        Args:
            api_key: API key for LLM provider
            model: Root LLM model name
            recursive_model: Sub-LLM model name for recursive calls
            max_iterations: Maximum number of root LLM iterations
            max_output_length: Maximum length of REPL output before truncation
            depth: Current recursion depth (0 for root)
        """
        self.api_key = api_key
        self.model = model
        self.recursive_model = recursive_model
        self.max_iterations = max_iterations
        self.max_output_length = max_output_length
        self.depth = depth

        self.llm = get_llm_client(api_key, model)
        self.sub_llm = get_llm_client(api_key, recursive_model)

        # Initialize cost tracking
        self._root_llm_cost = 0.0
        self._sub_llm_cost = 0.0
        self._root_llm_tokens = 0
        self._sub_llm_tokens = 0
        self._root_llm_calls = 0
        self._sub_llm_calls = 0

        # State
        self.repl_env: Optional[REPLEnv] = None
        self.messages: List[Dict[str, str]] = []
        self.query: Optional[str] = None
        self._context_debug: Dict[str, Any] = {}
        self._sub_llm_events: List[Dict[str, Any]] = []

    def _setup_context(
        self, context: Union[List[str], str, List[Dict[str, str]]], query: str
    ):
        """Setup the REPL environment with context."""
        logger.step("1", "SETTING UP CONTEXT & REPL ENVIRONMENT")

        # print("📝 What's happening in this step:")
        # print(
        #     "   - Context will be stored in REPL as 'context' variable (NOT sent to LLM!)"
        # )
        # print("   - REPL environment created with Python interpreter")
        # print("   - 'llm_query()' function injected for recursive sub-LLM calls")
        # print("   - 'FINAL_VAR()' function injected to return answers")

        self.query = query
        self.messages = []

        self.messages = build_system_prompt(self.model)

        # ==================Converting context for REPL supported format============
        context_data, context_str = self._convert_context(context)

        context_type, context_lengths, context_total_length = (
            self._get_context_metadata(context, context_data, context_str)
        )

        self._context_debug = {
            "context_type": context_type,
            "context_lengths": context_lengths,
            "context_total_length": context_total_length,
            "query": query,
        }

        def llm_query_fn(prompt: str) -> str:
            return self._recursive_llm_call(prompt)

        print("   Initializing REPL environment...")

        self.repl_env = REPLEnv(
            llm_query_fn=llm_query_fn,
            context_json=context_data,
            context_str=context_str,
        )

        # Adding context metadata to system prompt
        self.messages = add_context_metadata(
            self.messages, context_type, context_lengths, context_total_length
        )

        print(f"      - Context type: {context_type}")
        print(f"      - Context length: {context_total_length} characters")
        print(f"      - Variables in REPL: {list(self.repl_env.locals.keys())}")

    def _convert_context(self, context):
        """Convert context to appropriate format for REPL."""
        if isinstance(context, dict):
            return context, None
        elif isinstance(context, str):
            return None, context
        elif isinstance(context, list):
            if len(context) > 0 and isinstance(context[0], dict):
                if "content" in context[0]:
                    return [msg.get("content", "") for msg in context], None
                else:
                    return context, None
            else:
                return context, None
        else:
            return context, None

    def _get_context_metadata(self, context, context_data, context_str):
        """Get metadata about context for prompting."""
        if context_str is not None:
            context_type = "str"
            context_total_length = len(context_str)
            context_lengths = [context_total_length]
        elif context_data is not None:
            if isinstance(context_data, list):
                context_type = "list"
                context_lengths = [len(str(item)) for item in context_data]
                context_total_length = sum(context_lengths)
            elif isinstance(context_data, dict):
                context_type = "dict"
                context_lengths = [len(str(context_data))]
                context_total_length = context_lengths[0]
            else:
                context_type = type(context_data).__name__
                context_lengths = [len(str(context_data))]
                context_total_length = context_lengths[0]
        else:
            context_type = "unknown"
            context_lengths = [0]
            context_total_length = 0

        return context_type, context_lengths, context_total_length

    def _safe_value_preview(self, value: Any, max_chars: int = 400) -> str:
        """Return a safe short preview string for REPL variable values."""
        try:
            preview = repr(value)
        except Exception:
            preview = f"<{type(value).__name__}>"

        if len(preview) > max_chars:
            return preview[:max_chars] + f"... [truncated {len(preview)} chars]"
        return preview

    def _build_repl_state(self) -> Dict[str, Any]:
        """Build a clean REPL state snapshot for tracing and UI."""
        if not self.repl_env:
            return {}

        visible_locals = {}
        for key, value in self.repl_env.locals.items():
            if key.startswith("_"):
                continue
            visible_locals[key] = self._safe_value_preview(value)

        visible_globals = [
            k for k in self.repl_env.globals.keys() if not k.startswith("_")
        ]

        return {
            "context_loaded": "context" in self.repl_env.locals,
            "local_vars": list(visible_locals.keys()),
            "local_var_values": visible_locals,
            "globals": visible_globals,
        }

    def _build_root_input_snapshot(
        self, messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Build an exact snapshot of what is sent to the root LLM in a turn."""
        prior_sub_llm_responses = [
            str(event.get("response", ""))
            for event in self._sub_llm_events
            if event.get("response")
        ]

        message_summaries = []
        for i, msg in enumerate(messages, start=1):
            content = str(msg.get("content", ""))
            contains_sub_llm_output = any(
                resp and resp in content for resp in prior_sub_llm_responses
            )
            message_summaries.append(
                {
                    "index": i,
                    "role": msg.get("role", "unknown"),
                    "chars": len(content),
                    "contains_repl_output": "REPL output:" in content,
                    "contains_sub_llm_output": contains_sub_llm_output,
                    "content": content,
                    "content_preview": content[:6000],
                    "preview_truncated": len(content) > 6000,
                }
            )

        return {
            "message_count": len(messages),
            "total_chars": sum(item["chars"] for item in message_summaries),
            "messages": message_summaries,
            "contains_repl_feedback": any(
                item["contains_repl_output"] for item in message_summaries
            ),
            "contains_sub_llm_feedback": any(
                item["contains_sub_llm_output"] for item in message_summaries
            ),
        }

    def _recursive_llm_call(self, prompt: str) -> str:
        """
        Make a recursive LLM call (call the sub-LLM).
        """

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        response, cost_info = self.sub_llm.completion_with_cost(messages)

        self._sub_llm_cost += cost_info["cost"]
        self._sub_llm_tokens += cost_info["tokens"]
        self._sub_llm_calls += 1

        self._sub_llm_events.append(
            {
                "call_index": self._sub_llm_calls,
                "model": self.recursive_model,
                "prompt": prompt if isinstance(prompt, str) else str(prompt),
                "response": response,
                "tokens": cost_info.get("tokens", 0),
                "cost": cost_info.get("cost", 0.0),
            }
        )

        return response

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape codes from captured output for cleaner UI display."""
        ansi_escape_pattern = r"\x1B\[[0-?]*[ -/]*[@-~]"
        return re.sub(ansi_escape_pattern, "", text or "")

    def _find_code_blocks(self, text: str) -> Optional[List[str]]:
        """Find REPL code blocks in response."""
        pattern = r"```repl\s*\n(.*?)\n```"
        results = []

        for match in re.finditer(pattern, text, re.DOTALL):
            code_content = match.group(1).strip()
            results.append(code_content)

        return results if results else None

    def _find_final_answer(self, text: str) -> Optional[tuple]:
        """Find FINAL() or FINAL_VAR() in response."""
        # Check for FINAL_VAR first
        final_var_pattern = r"^\s*FINAL_VAR\((.*?)\)"
        match = re.search(final_var_pattern, text, re.MULTILINE | re.DOTALL)
        if match:
            return ("FINAL_VAR", match.group(1).strip())

        # Check for FINAL
        final_pattern = r"^\s*FINAL\((.*?)\)"
        match = re.search(final_pattern, text, re.MULTILINE | re.DOTALL)
        if match:
            return ("FINAL", match.group(1).strip())

        return None

    def _execute_code(self, code: str) -> str:
        """Execute code in REPL and return formatted result."""
        result = self.repl_env.code_execution(code)

        result_parts = []

        if result.stdout:
            result_parts.append(f"\n{self._strip_ansi(result.stdout)}")

        if result.stderr:
            result_parts.append(f"\nError: {self._strip_ansi(result.stderr)}")

        formatted = "\n".join(result_parts) if result_parts else "No output"

        # Truncate if too long - configurable truncation to align with paper
        # The paper mentions truncated outputs but doesn't specify length
        # Using a larger value to reduce impact on long outputs
        max_length = getattr(self, "max_output_length", 500000)
        if len(formatted) > max_length:
            formatted = (
                formatted[:max_length] + f"... [truncated from {len(formatted)} chars]"
            )

        return formatted

    def _process_code_execution(self, response: str) -> List[Dict[str, str]]:
        """Process code blocks in response and update messages."""
        messages, _ = self._process_code_execution_with_results(response)
        return messages

    def _process_code_execution_with_results(self, response: str):
        """Process code blocks in response and update messages, returning both messages and execution results."""
        code_blocks = self._find_code_blocks(response)
        execution_results = []

        if code_blocks:
            for code in code_blocks:
                execution_result = self._execute_code(code)
                execution_results.append(execution_result)

                # Add to messages
                self.messages.append(
                    {
                        "role": "user",
                        "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{execution_result}",
                    }
                )

        return self.messages, execution_results

    def _check_final_answer(self, response: str) -> Optional[str]:
        """Check if response contains final answer."""
        result = self._find_final_answer(response)
        if result is None:
            return None

        answer_type, content = result

        if answer_type == "FINAL":
            return content
        elif answer_type == "FINAL_VAR":
            # Get variable from REPL
            variable_name = (
                content.strip().strip('"').strip("'").strip("\n").strip("\r")
            )

            if variable_name in self.repl_env.locals:
                return str(self.repl_env.locals[variable_name])
            else:
                return None

        return None

    def completion(
        self, context: Union[List[str], str, List[Dict[str, str]]], query: str
    ) -> str:
        """
        Generate completion using RLM with REPL.

        Args:
            context: Context to process (can be arbitrarily long)
            query: Query to answer

        Returns:
            Final answer string
        """

        print("📋 Query:", query)
        print("🤖 Model:", self.model)
        print("🔄 Max iterations:", self.max_iterations)
        print()
        # print("📝 HOW RLM WORKS (different from normal LLM):")
        # print("   1. Context is stored OUTSIDE the LLM (in REPL 'context' variable)")
        # print("   2. LLM writes ```repl``` code to access/analyze context")
        # print("   3. Code executes in Python REPL, results fed back to LLM")
        # print("   4. LLM can call llm_query() for recursive sub-LLM calls")
        # print("   5. Loop continues until LLM returns FINAL(answer)")

        self._setup_context(context, query)

        logger.step("2", "STARTING ITERATIVE REASONING LOOP")
        # print("📝 What happens in each iteration:")
        # print("   a) Build prompt asking LLM what to do next")
        # print("   b) Call ROOT LLM to get response")
        # print("   c) Look for ```repl``` code blocks in response")
        # print("   d) Execute code blocks in REPL environment")
        # print("   e) Check if LLM returned FINAL() answer")
        # print("   f) If no FINAL(), continue to next iteration")

        # ================Main iteration loop===============
        for iteration in range(self.max_iterations):
            logger.separator(f">ITERATION {iteration + 1} / {self.max_iterations}", "=")
            sub_events_start = len(self._sub_llm_events)

            # =================Build user prompt for each iteration=============
            user_prompt = next_action_prompt(query, iteration)
            root_input_messages = self.messages + [user_prompt]
            root_input_snapshot = self._build_root_input_snapshot(root_input_messages)

            # logger.box("Prompt sent to LLM", user_prompt["content"], max_lines=8)

            logger.separator(f"Calling ROOT LLM ({self.model})", "")
            # print(f"   Sending {len(self.messages) + 1} messages to LLM...")
            response, cost_info = self.llm.completion_with_cost(root_input_messages)
            print(f"   Response received! ({cost_info['tokens']} tokens)")

            self._root_llm_cost += cost_info["cost"]
            self._root_llm_tokens += cost_info["tokens"]
            self._root_llm_calls += 1

            # print(f"Response received! ({cost_info['tokens']} tokens)")
            logger.box("LLM Response", response)

            logger.separator("LOOKING FOR ```repl``` CODE BLOCKS", "-", color="cyan")
            code_blocks = self._find_code_blocks(response)

            execution_results = []
            if code_blocks:
                print(f"   Found {len(code_blocks)} code block(s)")

                # logger.separator("EXECUTING CODE IN REPL", "-", color="green")
                # print("   📝 What's happening:")
                # print("      - Code runs in sandboxed Python REPL")
                # print("      - Can access 'context' variable (the stored context!)")
                # print("      - Can call 'llm_query()' for sub-LLM calls")
                # print()

                for i, code in enumerate(code_blocks):
                    logger.box(f"Code Block #{i+1} being executed", code)

                    execution_result = self._execute_code(code)
                    execution_results.append(execution_result)

                    logger.box(
                        f"REPL Output (from code block #{i+1})", execution_result
                    )

                    # Build variables content for log box
                    vars_content = []
                    for var, value in self.repl_env.locals.items():
                        if not var.startswith("_"):
                            vars_content.append(f"{var} = {value}")

                    if vars_content:
                        logger.box(
                            "Variables in REPL after execution", "\n".join(vars_content)
                        )

                    self.messages.append(
                        {
                            "role": "user",
                            "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{execution_result}",
                        }
                    )
            else:
                print("   ❌ No ```repl``` code blocks found in LLM response")
                print("   📝 Adding response as assistant message to conversation")
                self.messages.append(
                    {"role": "assistant", "content": "You responded with:\n" + response}
                )

            repl_state = self._build_repl_state()

            tracer.log_turn(
                iteration=iteration,
                messages=self.messages,
                response=response,
                code_blocks=code_blocks or [],
                execution_results=execution_results,
                repl_state=repl_state,
                cost_info=cost_info,
                sub_llm_calls=self._sub_llm_events[sub_events_start:],
                root_llm_input=root_input_snapshot,
            )

            logger.separator("CHECKING FOR FINAL() or FINAL_VAR()", "-", color="yellow")
            final_answer = self._check_final_answer(response)

            if final_answer:
                logger.separator("✅ FINAL ANSWER FOUND!", "=")
                logger.box("Final Answer", final_answer, max_lines=50)

                print("\n💰 COST SUMMARY:")
                print(f"- Root LLM calls: {self._root_llm_calls}")
                print(f"- Sub-LLM calls: {self._sub_llm_calls}")
                print(f"- Root LLM tokens: {self._root_llm_tokens}")
                print(f"- Sub-LLM tokens: {self._sub_llm_tokens}")
                print(f"- Total tokens: {self._root_llm_tokens + self._sub_llm_tokens}")
                print(
                    f"- Estimated cost: ${self._root_llm_cost + self._sub_llm_cost:.6f}"
                )

            if final_answer:

                tracer.log_turn(
                    iteration=iteration,
                    messages=self.messages,
                    response=response,
                    code_blocks=code_blocks or [],
                    execution_results=execution_results,
                    final_answer=final_answer,
                    repl_state=self._build_repl_state(),
                    cost_info=cost_info,
                    sub_llm_calls=self._sub_llm_events[sub_events_start:],
                    root_llm_input=root_input_snapshot,
                )
                return final_answer

            # Also check if any variable in the REPL environment contains a final answer
            # This handles cases where code execution created a variable with the answer
            if self.repl_env and hasattr(self.repl_env, "locals"):
                for var_name, var_value in self.repl_env.locals.items():
                    if isinstance(var_value, str):
                        # Check if this variable value is a final answer
                        if var_value.startswith("FINAL(") and var_value.endswith(")"):
                            actual_answer = var_value[
                                6:-1
                            ]  # Extract content between FINAL(...)
                            tracer.log_turn(
                                iteration=iteration,
                                messages=self.messages,
                                response=response,
                                code_blocks=code_blocks or [],
                                execution_results=execution_results,
                                final_answer=actual_answer,
                                repl_state=self._build_repl_state(),
                                cost_info=cost_info,
                                sub_llm_calls=self._sub_llm_events[sub_events_start:],
                                root_llm_input=root_input_snapshot,
                            )
                            return actual_answer

            # No final answer found this iteration
            print("   ❌ No FINAL() found yet")
            print(f"   ➡️Continuing to iteration {iteration + 2}...")

        # If no final answer after max iterations, return None to indicate timeout
        logger.separator("⚠️ MAX ITERATIONS REACHED", "!")
        print(
            f"RLM completed {self.max_iterations} iterations without finding FINAL() answer."
        )
        print("\n💰 COST SUMMARY:")
        print(f"   - Root LLM calls: {self._root_llm_calls}")
        print(f"   - Sub-LLM calls: {self._sub_llm_calls}")
        print(f"   - Total tokens: {self._root_llm_tokens + self._sub_llm_tokens}")

        # Log the timeout
        tracer.log_turn(
            iteration=self.max_iterations,
            messages=self.messages,
            response="",
            code_blocks=[],
            execution_results=[],
            final_answer=None,
            repl_state=self._build_repl_state(),
            cost_info={"cost": 0, "tokens": 0},
            sub_llm_calls=[],
            root_llm_input={},
        )

        return None

    def cost_summary(self) -> Dict[str, Any]:
        """Get cost summary for this completion."""
        return {
            "total_cost": self._root_llm_cost + self._sub_llm_cost,
            "root_llm_cost": self._root_llm_cost,
            "sub_llm_cost": self._sub_llm_cost,
            "root_llm_tokens": self._root_llm_tokens,
            "sub_llm_tokens": self._sub_llm_tokens,
            "root_llm_calls": self._root_llm_calls,
            "sub_llm_calls": self._sub_llm_calls,
        }

    def reset(self):
        """Reset RLM state."""
        self.repl_env = None
        self.messages = []
        self.query = None
        self._root_llm_cost = 0.0
        self._sub_llm_cost = 0.0
        self._root_llm_tokens = 0
        self._sub_llm_tokens = 0
        self._root_llm_calls = 0
        self._sub_llm_calls = 0
        self._context_debug = {}
        self._sub_llm_events = []

    def debug_summary(self) -> Dict[str, Any]:
        """Get debug metadata for current completion execution."""
        return {
            "context": self._context_debug,
            "models": {
                "root_model": self.model,
                "sub_model": self.recursive_model,
            },
            "root_prompt_stats": {
                "message_count": len(self.messages),
                "chars_total": sum(
                    len(str(m.get("content", ""))) for m in self.messages
                ),
            },
            "sub_llm_event_count": len(self._sub_llm_events),
        }
