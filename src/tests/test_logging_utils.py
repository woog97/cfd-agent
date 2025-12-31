"""
Tests for logging_utils module.
"""

import logging
import sys
import os

import pytest

# Add src2 to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_utils import (
    get_logger,
    configure_logging,
    set_verbose,
    LoggerAdapter,
)


class TestGetLogger:
    """Test get_logger function."""

    def test_returns_logger(self):
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_prefixed(self):
        logger = get_logger("mymodule")
        assert logger.name == "chatcfd.mymodule"

    def test_strips_src2_prefix(self):
        logger = get_logger("src2.database")
        assert logger.name == "chatcfd.database"

    def test_multiple_calls_same_logger(self):
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")
        assert logger1 is logger2


class TestConfigureLogging:
    """Test configure_logging function."""

    def test_configures_once(self):
        # configure_logging should be idempotent
        # Multiple calls should not add multiple handlers
        root = logging.getLogger("chatcfd")
        initial_handlers = len(root.handlers)

        configure_logging()
        configure_logging()
        configure_logging()

        # Should have at most initial + 1 handlers
        assert len(root.handlers) <= initial_handlers + 1


class TestSetVerbose:
    """Test set_verbose function."""

    def test_set_verbose_true(self):
        set_verbose(True)
        root = logging.getLogger("chatcfd")
        assert root.level == logging.DEBUG

    def test_set_verbose_false(self):
        set_verbose(False)
        root = logging.getLogger("chatcfd")
        assert root.level == logging.INFO


class TestLoggerAdapter:
    """Test LoggerAdapter class."""

    def test_callable_logs_info(self, caplog):
        logger = get_logger("test_adapter")
        adapter = LoggerAdapter(logger, verbose=True)

        with caplog.at_level(logging.INFO, logger="chatcfd.test_adapter"):
            adapter("Test message")

        assert "Test message" in caplog.text

    def test_callable_respects_verbose_false(self, caplog):
        logger = get_logger("test_adapter_quiet")
        adapter = LoggerAdapter(logger, verbose=False)

        with caplog.at_level(logging.INFO, logger="chatcfd.test_adapter_quiet"):
            adapter("Should not appear")

        assert "Should not appear" not in caplog.text

    def test_warning_always_logs(self, caplog):
        logger = get_logger("test_adapter_warning")
        adapter = LoggerAdapter(logger, verbose=False)

        with caplog.at_level(logging.WARNING, logger="chatcfd.test_adapter_warning"):
            adapter.warning("Warning message")

        assert "Warning message" in caplog.text

    def test_error_always_logs(self, caplog):
        logger = get_logger("test_adapter_error")
        adapter = LoggerAdapter(logger, verbose=False)

        with caplog.at_level(logging.ERROR, logger="chatcfd.test_adapter_error"):
            adapter.error("Error message")

        assert "Error message" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
