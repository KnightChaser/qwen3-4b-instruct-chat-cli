# io_utils.py
from __future__ import annotations

import sys


def safe_input(prompt: str = "") -> str:
    """
    Read a line from stdin without crashing on invalid UTF-8.

    - Writes `prompt` to stdout (no newline).
    - Reads raw bytes from sys.stdin.buffer.
    - Decodes with the current stdin encoding (or UTF-8 as fallback),
      replacing malformed sequences with the replacement char '�'.

    Args:
        prompt: Prompt string to display.

    Returns:
        The input string without trailing newlines.
    """
    # Show prompt
    sys.stdout.write(prompt)
    sys.stdout.flush()

    # Read raw bytes (no decoding yet)
    data = sys.stdin.buffer.readline()

    if not data:
        raise EOFError

    encoding = getattr(sys.stdin, "encoding", None) or "utf-8"

    try:
        return data.decode(encoding, errors="strict").rstrip("\n\r")
    except UnicodeDecodeError:
        # Fallback with replacement with debug message
        sys.stderr.write("[warn] Invalid bytes in input; replacing malformed characters.\n");
        sys.stderr.flush()
        return data.decode(encoding, errors="replace").rstrip("\n\r")
