"""
LLM client wrapper with cost tracking.
"""

from typing import Optional, Dict, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from config import settings


def get_llm_client(api_key: Optional[str] = None, model: str = None):
    """Factory function to create LLM client instances."""
    return LLMClient(
        api_key=api_key or settings.llm.api_key, model=model or settings.llm.model_name
    )


class LLMClient:
    """LLM client with cost tracking and timeout handling."""

    # Pricing per 1M tokens for NVIDIA models (input, output)
    PRICING = {
        "gpt-5": (2.50, 10.00),
        "gpt-5-mini": (0.15, 0.60),
        "gpt-5-nano": (0.10, 0.40),
    }

    def __init__(self, api_key: Optional[str] = None, model: str = None):
        """
        Initialize NVIDIA LLM client.

        Args:
            api_key: Optional API key (uses settings if not provided)
            model: Model name to use
        """
        try:
            self.base_url = settings.llm.base_url
            self.api_key = api_key or settings.llm.api_key
            self.model = model or settings.llm.model_name

            if not self.api_key:
                raise ValueError("NVIDIA_API_KEY not found in environment variables")

            if not self.base_url:
                raise ValueError("NVIDIA_BASE_URL not found in environment variables")

            self.client = ChatOpenAI(
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                temperature=settings.llm.temperature,
                request_timeout=settings.llm.request_timeout,
            )
            print(
                f"Successfully initialized NVIDIA LLM client with model: {self.model}"
            )

        except Exception as e:
            raise ValueError(f"Failed to initialize NVIDIA LLM client: {e}")

    def completion(self, messages, **kwargs) -> str:
        """Generate completion using NVIDIA model via ChatOpenAI."""
        try:
            if isinstance(messages, list) and all(
                isinstance(m, dict) for m in messages
            ):

                langchain_messages = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")

                    if role == "system":
                        langchain_messages.append(SystemMessage(content=content))
                    elif role == "assistant":
                        langchain_messages.append(AIMessage(content=content))
                    else:  # user
                        langchain_messages.append(HumanMessage(content=content))
            else:
                langchain_messages = [HumanMessage(content=str(messages))]

            client = ChatOpenAI(
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                temperature=kwargs.get("temperature", 0.7),
                request_timeout=120,  # 2 minute timeout
            )

            # ==================LLM Response===============
            response = client.invoke(langchain_messages)
            content = response.content

            return content

        except Exception as e:
            print(f"Error in completion method: {e}")
            import traceback

            traceback.print_exc()
            raise

    def completion_with_cost(self, messages, **kwargs) -> Tuple[str, Dict[str, float]]:
        """Completion with cost tracking via NVIDIA models."""
        try:
            content = self.completion(messages, **kwargs)

            # Calculate cost - estimate token counts
            input_tokens = (
                sum(
                    len(msg.get("content", ""))
                    for msg in messages
                    if isinstance(msg, dict)
                )
                // 4
            )  # Rough estimate
            output_tokens = len(content) // 4  # Rough estimate
            total_tokens = input_tokens + output_tokens

            # Get pricing for the model
            if self.model in self.PRICING:
                input_price, output_price = self.PRICING[self.model]
            else:
                # Default pricing for NVIDIA models
                input_price, output_price = 0.15, 0.60

            cost = (
                input_tokens / 1_000_000 * input_price
                + output_tokens / 1_000_000 * output_price
            )

            cost_info = {
                "cost": cost,
                "tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

            return content, cost_info

        except Exception as e:
            print(f"Error in completion_with_cost method: {e}")
            import traceback

            traceback.print_exc()
            raise

    def _make_request(self, messages, **kwargs):
        """
        Legacy method for backward compatibility.
        Delegates to completion method.
        """
        return self.completion(messages, **kwargs)
