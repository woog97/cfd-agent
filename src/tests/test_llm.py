"""
Tests for llm.py module.

Tests:
- LLMClient
- LLMResponse
- LLMStats
- ConversationManager
- estimate_tokens function
"""

import os
import sys
import json
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LLMConfig


# =============================================================================
# Test LLMResponse
# =============================================================================

class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_create_basic_response(self):
        """Can create LLMResponse with content."""
        from llm import LLMResponse

        response = LLMResponse(content="Test response")

        assert response.content == "Test response"
        assert response.reasoning_content == ""
        assert response.prompt_tokens == 0
        assert response.completion_tokens == 0
        assert response.reasoning_tokens == 0

    def test_create_full_response(self):
        """Can create LLMResponse with all fields."""
        from llm import LLMResponse

        response = LLMResponse(
            content="Test response",
            reasoning_content="Reasoning text",
            prompt_tokens=100,
            completion_tokens=50,
            reasoning_tokens=25,
        )

        assert response.content == "Test response"
        assert response.reasoning_content == "Reasoning text"
        assert response.prompt_tokens == 100
        assert response.completion_tokens == 50
        assert response.reasoning_tokens == 25


# =============================================================================
# Test LLMStats
# =============================================================================

class TestLLMStats:
    """Tests for LLMStats dataclass."""

    def test_initial_stats(self):
        """Stats start at zero."""
        from llm import LLMStats

        stats = LLMStats()

        assert stats.instruct_calls == 0
        assert stats.instruct_prompt_tokens == 0
        assert stats.instruct_completion_tokens == 0
        assert stats.reasoning_calls == 0
        assert stats.reasoning_prompt_tokens == 0
        assert stats.reasoning_completion_tokens == 0
        assert stats.reasoning_tokens == 0

    def test_record_instruct(self):
        """record_instruct updates stats correctly."""
        from llm import LLMStats

        stats = LLMStats()

        stats.record_instruct(100, 50)
        stats.record_instruct(200, 75)

        assert stats.instruct_calls == 2
        assert stats.instruct_prompt_tokens == 300
        assert stats.instruct_completion_tokens == 125

    def test_record_reasoning(self):
        """record_reasoning updates stats correctly."""
        from llm import LLMStats

        stats = LLMStats()

        stats.record_reasoning(100, 50, 25)
        stats.record_reasoning(200, 75, 30)

        assert stats.reasoning_calls == 2
        assert stats.reasoning_prompt_tokens == 300
        assert stats.reasoning_completion_tokens == 125
        assert stats.reasoning_tokens == 55


# =============================================================================
# Test estimate_tokens Function
# =============================================================================

class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    @pytest.mark.skipif(
        not pytest.importorskip("tiktoken", reason="tiktoken not installed"),
        reason="tiktoken required"
    )
    def test_estimate_tokens_short_text(self):
        """Estimates tokens for short text."""
        from llm import estimate_tokens

        # A simple sentence
        tokens = estimate_tokens("Hello, world!")

        assert tokens > 0
        assert tokens < 10  # Should be small

    @pytest.mark.skipif(
        not pytest.importorskip("tiktoken", reason="tiktoken not installed"),
        reason="tiktoken required"
    )
    def test_estimate_tokens_longer_text(self):
        """Longer text has more tokens."""
        from llm import estimate_tokens

        short = estimate_tokens("Hi")
        long = estimate_tokens("This is a much longer piece of text that should have more tokens.")

        assert long > short

    @pytest.mark.skipif(
        not pytest.importorskip("tiktoken", reason="tiktoken not installed"),
        reason="tiktoken required"
    )
    def test_estimate_tokens_empty(self):
        """Empty string has zero tokens."""
        from llm import estimate_tokens

        tokens = estimate_tokens("")

        assert tokens == 0


# =============================================================================
# Test LLMClient
# =============================================================================

class TestLLMClient:
    """Tests for LLMClient class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock LLMConfig."""
        return LLMConfig(
            base_url="https://test.api.com",
            instruct_model="test-instruct",
            reasoning_model="test-reasoning",
            instruct_temperature=0.5,
            reasoning_temperature=0.8,
        )

    @pytest.fixture
    def mock_openai(self):
        """Create a mock OpenAI client."""
        mock = Mock()
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="Test response"))]
        mock_completion.usage = Mock(prompt_tokens=10, completion_tokens=5)
        mock.chat.completions.create.return_value = mock_completion
        return mock

    def test_init(self, mock_config):
        """Can initialize LLMClient with config."""
        with patch('llm.OpenAI') as MockOpenAI:
            from llm import LLMClient

            client = LLMClient(mock_config)

            assert client.config == mock_config
            assert client.stats.instruct_calls == 0
            assert client.log_path is None

    def test_init_with_log_path(self, mock_config):
        """Can initialize with log path."""
        with patch('llm.OpenAI'):
            from llm import LLMClient

            client = LLMClient(mock_config, log_path="/tmp/llm.log")

            assert client.log_path == "/tmp/llm.log"

    def test_ask_instruct(self, mock_config, mock_openai):
        """ask_instruct calls API and returns response."""
        with patch('llm.OpenAI', return_value=mock_openai):
            from llm import LLMClient

            client = LLMClient(mock_config)
            response = client.ask_instruct("Test question")

            assert response == "Test response"
            assert client.stats.instruct_calls == 1
            mock_openai.chat.completions.create.assert_called_once()

    def test_ask_instruct_uses_correct_parameters(self, mock_config, mock_openai):
        """ask_instruct passes correct parameters to API."""
        with patch('llm.OpenAI', return_value=mock_openai):
            from llm import LLMClient

            client = LLMClient(mock_config)
            client.ask_instruct("Test question")

            call_kwargs = mock_openai.chat.completions.create.call_args[1]

            assert call_kwargs['model'] == "test-instruct"
            assert call_kwargs['temperature'] == 0.5
            assert call_kwargs['stream'] is False
            assert len(call_kwargs['messages']) == 1
            assert call_kwargs['messages'][0]['role'] == 'user'
            assert call_kwargs['messages'][0]['content'] == 'Test question'

    def test_ask_instruct_messages(self, mock_config, mock_openai):
        """ask_instruct_messages handles multi-turn conversation."""
        with patch('llm.OpenAI', return_value=mock_openai):
            from llm import LLMClient

            client = LLMClient(mock_config)

            messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "How are you?"},
            ]

            response = client.ask_instruct_messages(messages)

            assert response.content == "Test response"
            call_kwargs = mock_openai.chat.completions.create.call_args[1]
            assert len(call_kwargs['messages']) == 4

    def test_ask_instruct_updates_stats(self, mock_config, mock_openai):
        """ask_instruct updates statistics."""
        with patch('llm.OpenAI', return_value=mock_openai):
            from llm import LLMClient

            client = LLMClient(mock_config)
            client.ask_instruct("Question 1")
            client.ask_instruct("Question 2")

            assert client.stats.instruct_calls == 2
            assert client.stats.instruct_prompt_tokens == 20  # 10 + 10
            assert client.stats.instruct_completion_tokens == 10  # 5 + 5

    def test_ask_reasoning_streaming(self, mock_config):
        """ask_reasoning handles streaming responses."""
        mock_client = Mock()

        # Create mock streaming response
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock(delta=Mock(content="Part 1 ", model_extra={}))]
        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock(delta=Mock(content="Part 2", model_extra={}))]

        mock_client.chat.completions.create.return_value = iter([mock_chunk1, mock_chunk2])

        with patch('llm.OpenAI', return_value=mock_client):
            from llm import LLMClient

            client = LLMClient(mock_config)
            response = client.ask_reasoning("Test question")

            assert "Part 1" in response
            assert "Part 2" in response

    def test_logging(self, mock_config, mock_openai, tmp_path):
        """Client logs calls when log_path is set."""
        log_path = tmp_path / "llm.log"

        with patch('llm.OpenAI', return_value=mock_openai):
            from llm import LLMClient

            client = LLMClient(mock_config, log_path=str(log_path))
            client.ask_instruct("Test question")

            assert log_path.exists()
            with open(log_path) as f:
                log_content = f.read()
                log_data = json.loads(log_content)
                assert log_data['model_type'] == 'instruct'
                assert 'Test question' in str(log_data['messages'])


# =============================================================================
# Test ConversationManager
# =============================================================================

class TestConversationManager:
    """Tests for ConversationManager class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock LLMClient."""
        from llm import LLMResponse

        client = Mock()
        client.ask_instruct_messages.return_value = LLMResponse(content="Response")
        client.ask_reasoning_messages.return_value = LLMResponse(content="Reasoning response")
        return client

    def test_init_empty_history(self, mock_client):
        """ConversationManager starts with empty history."""
        from llm import ConversationManager

        conv = ConversationManager(mock_client, model_type="instruct")

        assert conv.history == []

    def test_init_with_system_prompt(self, mock_client):
        """ConversationManager can have system prompt."""
        from llm import ConversationManager

        conv = ConversationManager(
            mock_client,
            model_type="instruct",
            system_prompt="You are helpful"
        )

        assert len(conv.history) == 1
        assert conv.history[0]["role"] == "system"
        assert conv.history[0]["content"] == "You are helpful"

    def test_ask_maintains_history(self, mock_client):
        """ask() maintains conversation history."""
        from llm import ConversationManager

        conv = ConversationManager(mock_client, model_type="instruct")

        conv.ask("Question 1")

        assert len(conv.history) == 2  # user + assistant
        assert conv.history[0]["role"] == "user"
        assert conv.history[0]["content"] == "Question 1"
        assert conv.history[1]["role"] == "assistant"
        assert conv.history[1]["content"] == "Response"

    def test_ask_multiple_turns(self, mock_client):
        """Multiple asks accumulate in history."""
        from llm import ConversationManager, LLMResponse

        # Return different responses
        mock_client.ask_instruct_messages.side_effect = [
            LLMResponse(content="Response 1"),
            LLMResponse(content="Response 2"),
        ]

        conv = ConversationManager(mock_client, model_type="instruct")

        conv.ask("Question 1")
        conv.ask("Question 2")

        assert len(conv.history) == 4  # 2 user + 2 assistant
        assert conv.history[0]["content"] == "Question 1"
        assert conv.history[1]["content"] == "Response 1"
        assert conv.history[2]["content"] == "Question 2"
        assert conv.history[3]["content"] == "Response 2"

    def test_ask_uses_instruct_model(self, mock_client):
        """ask() uses instruct model when specified."""
        from llm import ConversationManager

        conv = ConversationManager(mock_client, model_type="instruct")
        conv.ask("Question")

        mock_client.ask_instruct_messages.assert_called_once()
        mock_client.ask_reasoning_messages.assert_not_called()

    def test_ask_uses_reasoning_model(self, mock_client):
        """ask() uses reasoning model when specified."""
        from llm import ConversationManager

        conv = ConversationManager(mock_client, model_type="reasoning")
        conv.ask("Question")

        mock_client.ask_reasoning_messages.assert_called_once()
        mock_client.ask_instruct_messages.assert_not_called()

    def test_clear_removes_history(self, mock_client):
        """clear() removes conversation history."""
        from llm import ConversationManager

        conv = ConversationManager(mock_client, model_type="instruct")
        conv.ask("Question")
        conv.clear()

        assert conv.history == []

    def test_clear_keeps_system_prompt(self, mock_client):
        """clear() keeps system prompt."""
        from llm import ConversationManager

        conv = ConversationManager(
            mock_client,
            model_type="instruct",
            system_prompt="You are helpful"
        )
        conv.ask("Question")
        conv.clear()

        assert len(conv.history) == 1
        assert conv.history[0]["role"] == "system"

    def test_get_history_returns_copy(self, mock_client):
        """get_history() returns a copy, not reference."""
        from llm import ConversationManager

        conv = ConversationManager(mock_client, model_type="instruct")
        conv.ask("Question")

        history_copy = conv.get_history()
        history_copy.append({"role": "user", "content": "Extra"})

        assert len(conv.history) == 2
        assert len(history_copy) == 3

    def test_passes_full_history_to_client(self, mock_client):
        """ask() passes full history to client."""
        from llm import ConversationManager, LLMResponse

        mock_client.ask_instruct_messages.return_value = LLMResponse(content="R")

        conv = ConversationManager(
            mock_client,
            model_type="instruct",
            system_prompt="System"
        )
        conv.ask("Q1")
        conv.ask("Q2")

        # Check the messages passed to the second call
        calls = mock_client.ask_instruct_messages.call_args_list
        second_call_messages = calls[1][0][0]

        assert len(second_call_messages) == 4  # system + user + assistant + user
        assert second_call_messages[0]["content"] == "System"
        assert second_call_messages[1]["content"] == "Q1"


# =============================================================================
# Integration Tests
# =============================================================================

class TestLLMIntegration:
    """Integration tests for LLM module."""

    def test_full_conversation_flow(self):
        """Test a full conversation flow with mocked client."""
        from llm import LLMClient, ConversationManager, LLMResponse

        config = LLMConfig(
            base_url="https://test.api.com",
            instruct_model="test",
            reasoning_model="test",
        )

        mock_openai = Mock()
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="I can help with that!"))]
        mock_completion.usage = Mock(prompt_tokens=10, completion_tokens=5)
        mock_openai.chat.completions.create.return_value = mock_completion

        with patch('llm.OpenAI', return_value=mock_openai):
            client = LLMClient(config)
            conv = ConversationManager(
                client,
                model_type="instruct",
                system_prompt="You are a CFD expert."
            )

            response1 = conv.ask("What is turbulence?")
            response2 = conv.ask("How does k-omega SST work?")

            assert len(conv.history) == 5  # system + 2*(user + assistant)
            assert client.stats.instruct_calls == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
