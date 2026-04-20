from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Protocol


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass
class GroqLLMClient:
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.0
    max_tokens: int = 512
    system_prompt: str = (
        "You are a precise assistant. Answer the user request directly and preserve "
        "the user's stated constraints."
    )

    def __post_init__(self) -> None:
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError(
                "The `groq` package is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        self._load_env_file()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to a real `.env` file or your shell environment. "
                "Do not put real keys in `.env.example`."
            )
        self.model = os.getenv("GROQ_MODEL", self.model)
        self._client = Groq(api_key=api_key)

    def _load_env_file(self) -> None:
        env_path = Path(__file__).resolve().parents[2] / ".env"
        try:
            from dotenv import load_dotenv
        except ImportError:
            return
        load_dotenv(dotenv_path=env_path)

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""
