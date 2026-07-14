import logging
from pathlib import Path

from src.utils.logger import get_logger


def test_get_logger_returns_logger_instance():
    logger = get_logger("test_logger_instance")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger_instance"


def test_get_logger_uses_info_level():
    logger = get_logger("test_logger_level")

    assert logger.level == logging.INFO
    assert logger.propagate is False


def test_get_logger_adds_console_and_file_handlers():
    logger = get_logger("test_logger_handlers")

    console_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
    ]

    file_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]

    assert len(console_handlers) == 1
    assert len(file_handlers) == 1
    assert len(logger.handlers) == 2


def test_get_logger_does_not_add_duplicate_handlers():
    logger_name = "test_logger_duplicate"

    first_logger = get_logger(logger_name)
    first_handler_ids = {
        id(handler)
        for handler in first_logger.handlers
    }

    second_logger = get_logger(logger_name)
    second_handler_ids = {
        id(handler)
        for handler in second_logger.handlers
    }

    assert first_logger is second_logger
    assert len(second_logger.handlers) == 2
    assert first_handler_ids == second_handler_ids


def test_logger_handler_levels_are_info():
    logger = get_logger("test_logger_handler_levels")

    for handler in logger.handlers:
        assert handler.level == logging.INFO


def test_logger_formatter_is_correct():
    logger = get_logger("test_logger_formatter")

    expected_format = (
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    )

    for handler in logger.handlers:
        assert handler.formatter is not None
        assert handler.formatter._fmt == expected_format


def test_logger_file_handler_targets_pipeline_log():
    logger = get_logger("test_logger_file_path")

    file_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]

    assert len(file_handlers) == 1

    log_path = Path(file_handlers[0].baseFilename)

    assert log_path.name == "pipeline.log"
    assert log_path.parent.name == "logs"
    assert log_path.parent.parent.name == "outputs"


def test_logger_writes_to_file():
    logger = get_logger("test_logger_file_write")

    test_message = "Logger file test message"

    logger.info(test_message)

    file_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]

    for handler in file_handlers:
        handler.flush()

    log_path = Path(file_handlers[0].baseFilename)

    assert log_path.exists()

    log_content = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    assert test_message in log_content