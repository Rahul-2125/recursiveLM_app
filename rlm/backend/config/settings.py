"""
Centralized configuration settings for RLM.
Loads environment variables and provides typed access to configuration.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for LLM clients."""

    base_url: str = field(default_factory=lambda: os.getenv("NVIDIA_BASE_URL", ""))
    api_key: str = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", ""))
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "NVIDIA_MODEL_NAME", "qwen/qwen2.5-coder-32b-instruct"
        )
    )
    temperature: float = 0.7
    request_timeout: int = 120  # seconds

    def validate(self) -> bool:
        """Validate required configuration is present."""
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not found in environment variables")
        if not self.base_url:
            raise ValueError("NVIDIA_BASE_URL not found in environment variables")
        return True


@dataclass
class RLMConfig:
    """Configuration for RLM engine."""

    max_iterations: int = 30
    max_output_length: int = 500000

    # Pricing per 1M tokens (input, output) for cost tracking
    pricing: dict = field(
        default_factory=lambda: {
            "gpt-5": (2.50, 10.00),
            "gpt-5-mini": (0.15, 0.60),
            "gpt-5-nano": (0.10, 0.40),
            "default": (0.15, 0.60),
        }
    )


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    log_dir: str = "logs"
    box_width: Optional[int] = None  # Auto-detect terminal width
    use_colors: bool = True
    timezone: str = "Asia/Kolkata"


@dataclass
class Settings:
    """Main settings container."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    rlm: RLMConfig = field(default_factory=RLMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> bool:
        """Validate all configuration."""
        return self.llm.validate()


settings = Settings()
