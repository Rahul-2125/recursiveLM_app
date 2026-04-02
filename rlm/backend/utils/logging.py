import logging
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional, Any

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

# IST timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))


def get_terminal_width() -> int:
    """Get terminal width, defaulting to 80 if not available."""
    try:
        return os.get_terminal_size().columns
    except:
        return 80


class Logger:
    COLORS = {
        "info": "\033[36m",  # Cyan
        "warning": "\033[33m",  # Yellow
        "error": "\033[91m",  # Red
        "success": "\033[92m",  # Green
        "header": "\033[95m",  # Magenta
        "bold": "\033[1m",
        "dim": "\033[2m",
        "cyan": "\033[36m",
        "green": "\033[92m",
        "yellow": "\033[33m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "reset": "\033[0m",
    }

    def __init__(self, box_width: int = None, use_colors: bool = True):
        self._box_width = box_width
        self.use_colors = use_colors and sys.stdout.isatty()

    @property
    def box_width(self) -> int:
        """Dynamic box width based on terminal size."""
        if self._box_width:
            return min(self._box_width, get_terminal_width() - 2)
        return min(100, get_terminal_width() - 2)

    def _color(self, text: str, color_name: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_colors:
            return text
        color = self.COLORS.get(color_name, "")
        reset = self.COLORS["reset"]
        return f"{color}{text}{reset}"

    def info(
        self,
        message: Optional[str] = None,
        origin: Optional[str] = None,
        data: Optional[dict] = None,
    ):
        self._log("info", message, origin, data)

    def error(
        self,
        message: Optional[str] = None,
        origin: Optional[str] = None,
        data: Optional[dict] = None,
    ):
        self._log("error", message, origin, data)

    def warning(
        self,
        message: Optional[str] = None,
        origin: Optional[str] = None,
        data: Optional[dict] = None,
    ):
        self._log("warning", message, origin, data)

    def success(
        self,
        message: Optional[str] = None,
        origin: Optional[str] = None,
        data: Optional[dict] = None,
    ):
        self._log("success", message, origin, data)

    def _log(self, level, message, origin, data):
        timestamp = datetime.now(IST).isoformat()
        color = self.COLORS.get(level, self.COLORS["reset"]) if self.use_colors else ""
        reset = self.COLORS["reset"] if self.use_colors else ""
        log_message = ""

        if origin:
            log_message += f"Origin: {origin} - "

        log_message += f"{color}{message}{reset}"

        if data:
            if isinstance(data, dict):
                log_message += f" - Data: {json.dumps(data, indent=4, default=str)}"
            else:
                log_message += f" - Data: {data}"

        log_level = level if level in ["info", "warning", "error"] else "info"
        getattr(logging, log_level)(log_message)

    # ============== BOX/SEPARATOR LOGGING ==============

    def separator(
        self, title: str, char: str = "=", width: int = None, color: str = "cyan"
    ):
        """Print a separator with a title."""
        w = width or min(80, get_terminal_width() - 2)
        line = char * w
        print(f"\n{self._color(line, color)}")
        print(f"  {self._color(title, color)}")
        print(f"{self._color(line, color)}\n")

    def step(self, step_num: str, description: str, color: str = "magenta"):
        """Print a step header."""
        w = min(80, get_terminal_width() - 2)
        line = "─" * w
        print(f"\n{self._color(line, color)}")
        print(self._color(f"📌 STEP {step_num}: {description}", color))
        print(f"{self._color(line, color)}")

    def box(self, title: str, content: Any, max_lines: int = None, color: str = "cyan"):
        """
        Print content in a box format.

        Args:
            title: Box title
            content: Content to display (will be converted to string)
            max_lines: If None, show all content. Otherwise limit to max_lines.
            color: Color for the box border
        """
        width = self.box_width
        inner_width = width - 4  # Account for "│ " and " │"

        # Box top
        print(self._color(f"┌{'─' * (width - 2)}┐", color))

        # Title
        title_display = title[:inner_width] if len(title) > inner_width else title
        print(
            self._color("│ ", color)
            + f"{title_display:<{inner_width}}"
            + self._color(" │", color)
        )

        # Separator
        print(self._color(f"├{'─' * (width - 2)}┤", color))

        # Content
        lines = str(content).split("\n")
        display_lines = lines if max_lines is None else lines[:max_lines]

        for line in display_lines:
            # Handle long lines by wrapping them
            while len(line) > inner_width:
                chunk = line[:inner_width]
                print(self._color("│ ", color) + chunk + self._color(" │", color))
                line = line[inner_width:]
            # Print remaining portion (padded)
            print(
                self._color("│ ", color)
                + f"{line:<{inner_width}}"
                + self._color(" │", color)
            )

        if max_lines is not None and len(lines) > max_lines:
            remaining = len(lines) - max_lines
            msg = f"... [{remaining} more lines]"
            print(
                self._color("│ ", color)
                + f"{msg:<{inner_width}}"
                + self._color(" │", color)
            )

        # Box bottom
        print(self._color(f"└{'─' * (width - 2)}┘", color))

    def header(self, text: str, emoji: str = "", color: str = "header"):
        """Print a styled header."""
        prefix = f"{emoji} " if emoji else ""
        color_code = (
            self.COLORS.get(color, self.COLORS["header"]) if self.use_colors else ""
        )
        bold = self.COLORS["bold"] if self.use_colors else ""
        reset = self.COLORS["reset"] if self.use_colors else ""
        print(f"\n{color_code}{bold}{prefix}{text}{reset}")

    def key_value(self, key: str, value: Any, indent: int = 0, color: str = None):
        """Print a key-value pair."""
        spaces = " " * indent
        if color and self.use_colors:
            color_code = self.COLORS.get(color, "")
            reset = self.COLORS["reset"]
            print(f"{spaces}{color_code}{key}{reset}: {value}")
        else:
            print(f"{spaces}{key}: {value}")

    def bullet(self, text: str, indent: int = 0, color: str = None):
        """Print a bullet point."""
        spaces = " " * indent
        if color and self.use_colors:
            color_code = self.COLORS.get(color, "")
            reset = self.COLORS["reset"]
            print(f"{spaces}{color_code}•{reset} {text}")
        else:
            print(f"{spaces}• {text}")

    def summary_box(self, title: str, items: dict, color: str = "cyan"):
        """Print a summary box with key-value pairs."""
        width = self.box_width
        inner_width = width - 4  # Account for "│ " and " │"

        print(self._color(f"┌{'─' * (width - 2)}┐", color))
        print(
            self._color("│ ", color)
            + f"{title:<{inner_width}}"
            + self._color(" │", color)
        )
        print(self._color(f"├{'─' * (width - 2)}┤", color))

        for key, value in items.items():
            line = f"{key}: {value}"
            # Wrap long lines
            while len(line) > inner_width:
                print(
                    self._color("│ ", color)
                    + f"{line[:inner_width]}"
                    + self._color(" │", color)
                )
                line = line[inner_width:]
            print(
                self._color("│ ", color)
                + f"{line:<{inner_width}}"
                + self._color(" │", color)
            )

        print(self._color(f"└{'─' * (width - 2)}┘", color))


logger = Logger()
