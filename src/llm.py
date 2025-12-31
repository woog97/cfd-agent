"""
LLM Client: Clean interface for language model interactions.

This module provides a dependency-injectable LLM client that doesn't rely on globals.
All configuration is passed explicitly.

Design principles:
- No global state
- Configuration passed via constructor
- Supports both instruct and reasoning models
- Clean message interface
- Optional logging to file
"""

import functools
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, TypeVar, Literal

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay between retries.
        exponential_base: Base for exponential backoff calculation.
        retryable_exceptions: Tuple of exception types to retry on.

    Usage:
        @retry_with_backoff(max_retries=3)
        def my_api_call():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt < max_retries - 1:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        print(f" [retry {attempt + 1}/{max_retries - 1} in {delay:.1f}s: {e.__class__.__name__}]", end="", flush=True)
                        time.sleep(delay)

            # All retries exhausted
            raise last_exception

        return wrapper
    return decorator

import tiktoken
from openai import OpenAI

from config import LLMConfig


@dataclass
class LLMResponse:
    """Structured response from LLM call."""
    content: str
    reasoning_content: str = ""  # For reasoning models that expose this
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class LLMStats:
    """Statistics for LLM usage tracking."""
    instruct_calls: int = 0
    instruct_prompt_tokens: int = 0
    instruct_completion_tokens: int = 0
    reasoning_calls: int = 0
    reasoning_prompt_tokens: int = 0
    reasoning_completion_tokens: int = 0
    reasoning_tokens: int = 0

    def record_instruct(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.instruct_calls += 1
        self.instruct_prompt_tokens += prompt_tokens
        self.instruct_completion_tokens += completion_tokens

    def record_reasoning(self, prompt_tokens: int, completion_tokens: int, reasoning_tokens: int = 0) -> None:
        self.reasoning_calls += 1
        self.reasoning_prompt_tokens += prompt_tokens
        self.reasoning_completion_tokens += completion_tokens
        self.reasoning_tokens += reasoning_tokens


def estimate_tokens(text: str) -> int:
    """Estimate token count using tiktoken cl100k_base encoding."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


class LLMClient:
    """
    Clean LLM client with explicit configuration.

    Usage:
        config = LLMConfig.from_env()
        client = LLMClient(config)

        # Single question (stateless)
        response = client.ask_instruct("What is CFD?")

        # With conversation history
        messages = [
            {"role": "user", "content": "What is CFD?"},
            {"role": "assistant", "content": "CFD stands for..."},
            {"role": "user", "content": "How is it used in engineering?"},
        ]
        response = client.ask_instruct_messages(messages)
    """

    def __init__(
        self,
        config: LLMConfig,
        log_path: str | None = None,
    ):
        """
        Initialize LLM client.

        Args:
            config: LLM configuration (models, temperatures, etc.)
            log_path: Optional path to write logs. If None, no logging.
        """
        self.config = config
        self.log_path = log_path
        self.stats = LLMStats()
        self._client = OpenAI(base_url=config.base_url)
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def ask_instruct(self, question: str) -> str:
        """
        Ask a single question using the instruct model (stateless).

        Args:
            question: The question to ask.

        Returns:
            The model's response content.
        """
        messages = [{"role": "user", "content": question}]
        response = self.ask_instruct_messages(messages)
        return response.content

    def ask_instruct_messages(self, messages: list[dict[str, str]]) -> LLMResponse:
        """
        Send messages to the instruct model.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            LLMResponse with content and token counts.
        """
        completion = self._call_instruct_api(messages)

        response = LLMResponse(
            content=completion.choices[0].message.content or "",
            prompt_tokens=completion.usage.prompt_tokens if completion.usage else 0,
            completion_tokens=completion.usage.completion_tokens if completion.usage else 0,
        )

        self.stats.record_instruct(response.prompt_tokens, response.completion_tokens)
        self._log("instruct", messages, response)

        return response

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _call_instruct_api(self, messages: list[dict[str, str]]):
        """Make the actual API call to instruct model (with retry)."""
        return self._client.chat.completions.create(
            messages=messages,
            model=self.config.instruct_model,
            temperature=self.config.instruct_temperature,
            stream=False,
        )

    def ask_reasoning(self, question: str) -> str:
        """
        Ask a single question using the reasoning model (stateless).

        Args:
            question: The question to ask.

        Returns:
            The model's response content.
        """
        messages = [{"role": "user", "content": question}]
        response = self.ask_reasoning_messages(messages)
        return response.content

    def ask_reasoning_messages(self, messages: list[dict[str, str]]) -> LLMResponse:
        """
        Send messages to the reasoning model (streaming).

        Handles DeepSeek-style reasoning_content if present.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            LLMResponse with content, reasoning_content, and token counts.
        """
        content, reasoning_content = self._call_reasoning_api_streaming(messages)

        # Estimate tokens since streaming doesn't provide usage
        prompt_str = json.dumps(messages, ensure_ascii=False)
        prompt_tokens = estimate_tokens(prompt_str)
        completion_tokens = estimate_tokens(content)
        reasoning_tokens = estimate_tokens(reasoning_content) if reasoning_content else 0

        response = LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
        )

        self.stats.record_reasoning(
            response.prompt_tokens,
            response.completion_tokens,
            response.reasoning_tokens,
        )
        self._log("reasoning", messages, response)

        return response

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _call_reasoning_api_streaming(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        """
        Make the actual streaming API call to reasoning model (with retry).

        Returns:
            Tuple of (content, reasoning_content).
        """
        stream = self._client.chat.completions.create(
            messages=messages,
            model=self.config.reasoning_model,
            temperature=self.config.reasoning_temperature,
            stream=True,
        )

        content_parts = []
        reasoning_parts = []

        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    content_parts.append(delta.content)
                # Handle reasoning content if present (DeepSeek-specific)
                if hasattr(delta, 'model_extra') and delta.model_extra:
                    if 'reasoning_content' in delta.model_extra:
                        reasoning_parts.append(str(delta.model_extra['reasoning_content']))

        return "".join(content_parts), "".join(reasoning_parts)

    def _log(
        self,
        model_type: Literal["instruct", "reasoning"],
        messages: list[dict[str, str]],
        response: LLMResponse,
    ) -> None:
        """Write log entry to file if log_path is set."""
        if not self.log_path:
            return

        log_entry = {
            "model_type": model_type,
            "messages": messages,
            "response": response.content,
            "reasoning_content": response.reasoning_content,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "reasoning_tokens": response.reasoning_tokens,
            "timestamp": datetime.now().isoformat(),
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')


class ConversationManager:
    """
    Manages conversation history for multi-turn interactions.

    Usage:
        client = LLMClient(config)
        conv = ConversationManager(client, model_type="reasoning")

        response1 = conv.ask("What is CFD?")
        response2 = conv.ask("How is it used?")  # Has context from previous exchange
    """

    def __init__(
        self,
        client: LLMClient,
        model_type: Literal["instruct", "reasoning"] = "reasoning",
        system_prompt: str | None = None,
    ):
        """
        Initialize conversation manager.

        Args:
            client: LLM client instance.
            model_type: Which model to use ("instruct" or "reasoning").
            system_prompt: Optional system prompt to prepend.
        """
        self.client = client
        self.model_type = model_type
        self.history: list[dict[str, str]] = []

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def ask(self, question: str) -> str:
        """
        Ask a question with full conversation history.

        Args:
            question: The question to ask.

        Returns:
            The model's response content.
        """
        self.history.append({"role": "user", "content": question})

        if self.model_type == "instruct":
            response = self.client.ask_instruct_messages(self.history.copy())
        else:
            response = self.client.ask_reasoning_messages(self.history.copy())

        self.history.append({"role": "assistant", "content": response.content})

        return response.content

    def clear(self) -> None:
        """Clear conversation history (keeps system prompt if set)."""
        system_msgs = [m for m in self.history if m["role"] == "system"]
        self.history = system_msgs

    def get_history(self) -> list[dict[str, str]]:
        """Get a copy of the conversation history."""
        return self.history.copy()
