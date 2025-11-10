# qwen3_chat.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Any

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


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
    max_tokens: int = 2048 # Maximum tokens to generate

@dataclass
class Qwen3ChatEngine:
    """
    Tiny wrapper around vLLM + Qwen3 for a single-user CLI chat.
    """
    config: Qwen3Config = field(default_factory=Qwen3Config)
    _llm: LLM | None = field(init=False, default=None)
    _tokenizer: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._llm = LLM(
            model=self.config.model_name,
            trust_remote_code=True,
            max_model_len=self.config.max_model_len,
        )

        # Initialize tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

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
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized.")

        # Convert our dataclass messages into HF-style dict messages
        hf_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        prompt: str = self._tokenizer.apply_chat_template(
            hf_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return prompt

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
