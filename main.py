# main.py
from __future__ import annotations

import sys
from typing import List

from qwen3_chat import Qwen3ChatEngine, Qwen3Config, ChatMessage
from io_utils import safe_input


def run_cli() -> None:
    print("Qwen3 CLI Chat (type /exit to quit)")
    print("Loading model... (first run may download weights)")

    engine = Qwen3ChatEngine(Qwen3Config())

    # System prompt: keep it short and neutral
    messages: List[ChatMessage] = [
        ChatMessage(
            role="system",
            content=(
                "You are a helpful, concise assistant. "
                "Answer in the same language as the user whenever possible."
            ),
        )
    ]

    print("Model loaded. Start chatting!\n")

    try:
        while True:
            try:
                user_input = safe_input("You: ").strip()
            except EOFError:
                print("\n[EOF] Exiting.")
                break

            if not user_input:
                continue

            if user_input.lower() in {"/exit", "/quit"}:
                print("Bye, bye! >_<")
                break

            messages.append(ChatMessage(role="user", content=user_input))

            # Generate a reply
            reply = engine.chat(messages)
            messages.append(ChatMessage(role="assistant", content=reply))

            # Print model response
            print(f"Qwen3: {reply}\n")

    except KeyboardInterrupt:
        print("\n[Ctrl-C] Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    run_cli()

