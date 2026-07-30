from datetime import date

import pytest

from dashboard.filters import (
    comparison_has_same_length,
    get_comparison_range,
    resolve_date_range,
)


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        (
            "today",
            (
                date(2026, 7, 29),
                date(2026, 7, 29),
            ),
        ),
        (
            "yesterday",
            (
                date(2026, 7, 28),
                date(2026, 7, 28),
            ),
        ),
        (
            "this_week",
            (
                date(2026, 7, 27),
                date(2026, 7, 29),
            ),
        ),
        (
            "last_week",
            (
                date(2026, 7, 20),
                date(2026, 7, 26),
            ),
        ),
        (
            "last_7_days",
            (
                date(2026, 7, 23),
                date(2026, 7, 29),
            ),
        ),
        (
            "last_30_days",
            (
                date(2026, 6, 30),
                date(2026, 7, 29),
            ),
        ),
        (
            "last_60_days",
            (
                date(2026, 5, 31),
                date(2026, 7, 29),
            ),
        ),
        (
            "last_90_days",
            (
                date(2026, 5, 1),
                date(2026, 7, 29),
            ),
        ),
        (
            "this_month",
            (
                date(2026, 7, 1),
                date(2026, 7, 29),
            ),
        ),
        (
            "last_month",
            (
                date(2026, 6, 1),
                date(2026, 6, 30),
            ),
        ),
        (
            "this_quarter",
            (
                date(2026, 7, 1),
                date(2026, 7, 29),
            ),
        ),
        (
            "last_quarter",
            (
                date(2026, 4, 1),
                date(2026, 6, 30),
            ),
        ),
        (
            "this_year",
            (
                date(2026, 1, 1),
                date(2026, 7, 29),
            ),
        ),
        (
            "last_year",
            (
                date(2025, 1, 1),
                date(2025, 12, 31),
            ),
        ),
    ],
)
def test_resolve_date_range(
    preset: str,
    expected: tuple[date, date],
) -> None:
    result = resolve_date_range(
        preset,
        today=date(2026, 7, 29),
    )

    assert result == expected


def test_unknown_preset_returns_last_30_days() -> None:
    result = resolve_date_range(
        "unknown_preset",
        today=date(2026, 7, 29),
    )

    assert result == (
        date(2026, 6, 30),
        date(2026, 7, 29),
    )


def test_previous_period_has_equal_length() -> None:
    current_start = date(2026, 7, 1)
    current_end = date(2026, 7, 30)

    comparison = get_comparison_range(
        start_date=current_start,
        end_date=current_end,
        comparison="previous_period",
    )

    assert comparison == (
        date(2026, 6, 1),
        date(2026, 6, 30),
    )

    assert comparison_has_same_length(
        start_date=current_start,
        end_date=current_end,
        comparison_start_date=comparison[0],
        comparison_end_date=comparison[1],
    )


def test_previous_period_for_single_day() -> None:
    comparison = get_comparison_range(
        start_date=date(2026, 7, 29),
        end_date=date(2026, 7, 29),
        comparison="previous_period",
    )

    assert comparison == (
        date(2026, 7, 28),
        date(2026, 7, 28),
    )


def test_previous_month_comparison() -> None:
    comparison = get_comparison_range(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 29),
        comparison="previous_month",
    )

    assert comparison == (
        date(2026, 6, 1),
        date(2026, 6, 30),
    )


def test_previous_month_across_year_boundary() -> None:
    comparison = get_comparison_range(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 20),
        comparison="previous_month",
    )

    assert comparison == (
        date(2025, 12, 1),
        date(2025, 12, 31),
    )


def test_previous_quarter_comparison() -> None:
    comparison = get_comparison_range(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 29),
        comparison="previous_quarter",
    )

    assert comparison == (
        date(2026, 4, 1),
        date(2026, 6, 30),
    )


def test_previous_quarter_across_year_boundary() -> None:
    comparison = get_comparison_range(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 2, 15),
        comparison="previous_quarter",
    )

    assert comparison == (
        date(2025, 10, 1),
        date(2025, 12, 31),
    )


def test_previous_year_comparison() -> None:
    comparison = get_comparison_range(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 29),
        comparison="previous_year",
    )

    assert comparison == (
        date(2025, 7, 1),
        date(2025, 7, 29),
    )


def test_previous_year_comparison_handles_leap_day() -> None:
    comparison = get_comparison_range(
        start_date=date(2024, 2, 29),
        end_date=date(2024, 3, 5),
        comparison="previous_year",
    )

    assert comparison == (
        date(2023, 2, 28),
        date(2023, 3, 5),
    )


def test_previous_year_ytd_comparison() -> None:
    comparison = get_comparison_range(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 7, 29),
        comparison="previous_year_ytd",
    )

    assert comparison == (
        date(2025, 1, 1),
        date(2025, 7, 29),
    )


def test_previous_year_ytd_handles_leap_day() -> None:
    comparison = get_comparison_range(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 2, 29),
        comparison="previous_year_ytd",
    )

    assert comparison == (
        date(2023, 1, 1),
        date(2023, 2, 28),
    )


def test_custom_comparison_sorts_reversed_dates() -> None:
    comparison = get_comparison_range(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 29),
        comparison="custom_comparison",
        custom_start_date=date(2026, 6, 30),
        custom_end_date=date(2026, 6, 1),
    )

    assert comparison == (
        date(2026, 6, 1),
        date(2026, 6, 30),
    )


def test_custom_comparison_requires_both_dates() -> None:
    comparison = get_comparison_range(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 29),
        comparison="custom_comparison",
        custom_start_date=date(2026, 6, 1),
        custom_end_date=None,
    )

    assert comparison is None


def test_no_comparison_returns_none() -> None:
    comparison = get_comparison_range(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 29),
        comparison="no_comparison",
    )

    assert comparison is None


def test_unknown_comparison_returns_none() -> None:
    comparison = get_comparison_range(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 29),
        comparison="unknown_comparison",
    )

    assert comparison is None


def test_comparison_length_validation_returns_true() -> None:
    result = comparison_has_same_length(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 30),
        comparison_start_date=date(2026, 6, 1),
        comparison_end_date=date(2026, 6, 30),
    )

    assert result is True


def test_comparison_length_validation_detects_mismatch() -> None:
    result = comparison_has_same_length(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 30),
        comparison_start_date=date(2026, 6, 1),
        comparison_end_date=date(2026, 6, 15),
    )

    assert result is False


def test_missing_comparison_dates_are_valid() -> None:
    result = comparison_has_same_length(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 30),
        comparison_start_date=None,
        comparison_end_date=None,
    )

    assert result is True