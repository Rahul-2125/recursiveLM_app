"""
RLM - Recursive Language Model

A Python implementation of Recursive Language Models that can process
arbitrarily long contexts by storing them externally in a REPL environment.
"""

__version__ = "1.0.0"
__author__ = "RLM Team"

from core import RLM, RLM_REPL, REPLEnv
from config import settings
from main import run

__all__ = [
    "RLM",
    "RLM_REPL", 
    "REPLEnv",
    "settings",
    "run",
    "__version__",
]