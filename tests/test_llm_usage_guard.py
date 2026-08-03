from __future__ import annotations

import json
from datetime import date, timedelta

import src.llm.usage_guard as usage_guard


def _configure_guard(
    monkeypatch,
    tmp_path,
    limit: int = 3,
):
    usage_file = tmp_path / "llm_usage.json"

    monkeypatch.setattr(
        usage_guard,
        "LLM_USAGE_FILE",
        str(usage_file),
    )

    monkeypatch.setattr(
        usage_guard,
        "LLM_DAILY_REQUEST_LIMIT",
        limit,
    )

    return usage_file


def test_initial_usage_is_zero(
    monkeypatch,
    tmp_path,
):
    _configure_guard(
        monkeypatch,
        tmp_path,
    )

    usage = usage_guard.get_daily_usage()

    assert usage["requests"] == 0
    assert usage["limit"] == 3
    assert usage["remaining"] == 3
    assert usage["limit_reached"] is False


def test_register_request_increments_usage(
    monkeypatch,
    tmp_path,
):
    _configure_guard(
        monkeypatch,
        tmp_path,
    )

    assert usage_guard.register_llm_request() is True

    usage = usage_guard.get_daily_usage()

    assert usage["requests"] == 1
    assert usage["remaining"] == 2
    assert usage["limit_reached"] is False


def test_daily_limit_blocks_extra_requests(
    monkeypatch,
    tmp_path,
):
    _configure_guard(
        monkeypatch,
        tmp_path,
        limit=2,
    )

    assert usage_guard.register_llm_request() is True
    assert usage_guard.register_llm_request() is True
    assert usage_guard.register_llm_request() is False

    usage = usage_guard.get_daily_usage()

    assert usage["requests"] == 2
    assert usage["remaining"] == 0
    assert usage["limit_reached"] is True
    assert usage_guard.llm_request_allowed() is False


def test_usage_resets_on_new_day(
    monkeypatch,
    tmp_path,
):
    usage_file = _configure_guard(
        monkeypatch,
        tmp_path,
    )

    yesterday = (
        date.today()
        - timedelta(days=1)
    ).isoformat()

    usage_file.write_text(
        json.dumps(
            {
                "date": yesterday,
                "requests": 3,
            }
        ),
        encoding="utf-8",
    )

    usage = usage_guard.get_daily_usage()

    assert usage["requests"] == 0
    assert usage["remaining"] == 3
    assert usage["limit_reached"] is False


def test_corrupt_usage_file_falls_back_safely(
    monkeypatch,
    tmp_path,
):
    usage_file = _configure_guard(
        monkeypatch,
        tmp_path,
    )

    usage_file.write_text(
        "not-valid-json",
        encoding="utf-8",
    )

    usage = usage_guard.get_daily_usage()

    assert usage["requests"] == 0
    assert usage["remaining"] == 3
    assert usage["limit_reached"] is False