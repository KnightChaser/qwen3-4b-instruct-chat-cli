# qwen3_chat.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

from vllm import LLM, SamplingParams


Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str


@dataclass
class Qwen3Config:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507" # fixed
    max_model_len: int = 32768
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512

@dataclass
class Qwen3ChatEngine:
    """
    Tiny wrapper around vLLM + Qwen3 for a single-user CLI chat.
    """
    config: Qwen3Config = field(default_factory=Qwen3Config)
    _llm: LLM | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._llm = LLM(
            model=self.config.model_name,
            trust_remote_code=True,
            max_model_len=self.config.max_model_len,
        )

    def _build_prompt(self, messages: List[ChatMessage]) -> str:
        """
        Build a plain text prompt for vLLM for chat message.

        Here we use a very simple template compatible with Qwen3:
        - System message at the top.
        - Then alternating user/assistant turns with clear markers.

        Args:
            messages: List of chat messages.

        Returns:
            The constructed prompt string.
        """
        system_prefix: str = ""
        remaining: List[ChatMessage] = messages

        if messages and messages[0].role == "system":
            system_prefix = f"System: {messages[0].content.strip()}\n\n"
            remaining = messages[1:]

        lines: List[str] = []
        for m in remaining:
            if m.role == "user":
                lines.append(f"User: {m.content.strip()}")
            elif m.role == "assistant":
                lines.append(f"Assistant: {m.content.strip()}")
            else:
                raise ValueError(f"Unexpected role: {m.role}, expected 'user' or 'assistant'.")

        full_prompt = system_prefix + "\n".join(lines) + "Assistant: "
        return full_prompt

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate a response from the model given the chat messages.

        Args:
            messages: List of chat messages.
            temperature: Sampling temperature. If None, use config default.
            max_tokens: Maximum tokens to generate. If None, use config default.

        Returns:
            The generated response text.
        """
        if self._llm is None:
            raise RuntimeError("LLM not initialized.")

        sampling_params = SamplingParams(
            temperature=temperature if temperature is not None else self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
        )
        prompt = self._build_prompt(messages)
        outputs = self._llm.generate([prompt], sampling_params)

        # We only send one prompt
        output = outputs[0]
        text = output.outputs[0].text

        return text.strip()
