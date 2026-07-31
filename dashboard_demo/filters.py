from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta

import streamlit as st

from dashboard_demo.i18n import translate


@dataclass(frozen=True)
class DashboardFilters:
    """Shared dashboard filter state."""

    language: str
    preset: str
    start_date: date
    end_date: date
    comparison: str
    comparison_start_date: date | None = None
    comparison_end_date: date | None = None
    comparison_same_length: bool = True


DATE_PRESET_KEYS = [
    "today",
    "yesterday",
    "this_week",
    "last_week",
    "last_7_days",
    "last_30_days",
    "last_60_days",
    "last_90_days",
    "this_month",
    "last_month",
    "this_quarter",
    "last_quarter",
    "this_year",
    "last_year",
    "custom_range",
]

COMPARISON_KEYS = [
    "no_comparison",
    "previous_period",
    "previous_month",
    "previous_quarter",
    "previous_year",
    "previous_year_ytd",
    "custom_comparison",
]


def _month_start(value: date) -> date:
    return value.replace(day=1)


def _previous_month_start(value: date) -> date:
    first_day = _month_start(value)
    return (first_day - timedelta(days=1)).replace(day=1)


def _quarter_start(value: date) -> date:
    quarter_month = ((value.month - 1) // 3) * 3 + 1
    return date(value.year, quarter_month, 1)


def _previous_quarter_start(value: date) -> date:
    current_start = _quarter_start(value)
    previous_quarter_end = current_start - timedelta(days=1)
    return _quarter_start(previous_quarter_end)


def _week_start(value: date) -> date:
    """Return Monday for the week containing value."""
    return value - timedelta(days=value.weekday())


def _safe_replace_year(value: date, year: int) -> date:
    """Move a date to another year without failing on 29 February."""
    day = min(
        value.day,
        monthrange(year, value.month)[1],
    )

    return value.replace(
        year=year,
        day=day,
    )


def _period_length(
    start_date: date,
    end_date: date,
) -> int:
    return (end_date - start_date).days + 1


def comparison_has_same_length(
    start_date: date,
    end_date: date,
    comparison_start_date: date | None,
    comparison_end_date: date | None,
) -> bool:
    """Return True when analysis and comparison periods have equal lengths."""
    if (
        comparison_start_date is None
        or comparison_end_date is None
    ):
        return True

    return _period_length(
        start_date,
        end_date,
    ) == _period_length(
        comparison_start_date,
        comparison_end_date,
    )


def resolve_date_range(
    preset: str,
    today: date | None = None,
) -> tuple[date, date]:
    """Resolve a preset into inclusive start and end dates."""
    current_date = today or date.today()

    if preset == "today":
        return current_date, current_date

    if preset == "yesterday":
        yesterday = current_date - timedelta(days=1)
        return yesterday, yesterday

    if preset == "this_week":
        return _week_start(current_date), current_date

    if preset == "last_week":
        current_week_start = _week_start(current_date)
        previous_week_end = (
            current_week_start - timedelta(days=1)
        )
        previous_week_start = (
            previous_week_end - timedelta(days=6)
        )

        return previous_week_start, previous_week_end

    if preset == "last_7_days":
        return (
            current_date - timedelta(days=6),
            current_date,
        )

    if preset == "last_30_days":
        return (
            current_date - timedelta(days=29),
            current_date,
        )

    if preset == "last_60_days":
        return (
            current_date - timedelta(days=59),
            current_date,
        )

    if preset == "last_90_days":
        return (
            current_date - timedelta(days=89),
            current_date,
        )

    if preset == "this_month":
        return (
            _month_start(current_date),
            current_date,
        )

    if preset == "last_month":
        start_date = _previous_month_start(current_date)
        end_date = (
            _month_start(current_date)
            - timedelta(days=1)
        )

        return start_date, end_date

    if preset == "this_quarter":
        return (
            _quarter_start(current_date),
            current_date,
        )

    if preset == "last_quarter":
        start_date = _previous_quarter_start(current_date)
        end_date = (
            _quarter_start(current_date)
            - timedelta(days=1)
        )

        return start_date, end_date

    if preset == "this_year":
        return (
            date(current_date.year, 1, 1),
            current_date,
        )

    if preset == "last_year":
        return (
            date(current_date.year - 1, 1, 1),
            date(current_date.year - 1, 12, 31),
        )

    return (
        current_date - timedelta(days=29),
        current_date,
    )


def get_comparison_range(
    start_date: date,
    end_date: date,
    comparison: str,
    custom_start_date: date | None = None,
    custom_end_date: date | None = None,
) -> tuple[date, date] | None:
    """Return an inclusive comparison period."""
    start_date = min(start_date, end_date)
    end_date = max(start_date, end_date)

    if comparison == "no_comparison":
        return None

    if comparison == "previous_period":
        period_days = _period_length(
            start_date,
            end_date,
        )

        comparison_end = (
            start_date - timedelta(days=1)
        )

        comparison_start = (
            comparison_end
            - timedelta(days=period_days - 1)
        )

        return (
            comparison_start,
            comparison_end,
        )

    if comparison == "previous_month":
        comparison_start = _previous_month_start(
            start_date
        )

        comparison_end = (
            _month_start(start_date)
            - timedelta(days=1)
        )

        return (
            comparison_start,
            comparison_end,
        )

    if comparison == "previous_quarter":
        comparison_start = _previous_quarter_start(
            start_date
        )

        comparison_end = (
            _quarter_start(start_date)
            - timedelta(days=1)
        )

        return (
            comparison_start,
            comparison_end,
        )

    if comparison == "previous_year":
        return (
            _safe_replace_year(
                start_date,
                start_date.year - 1,
            ),
            _safe_replace_year(
                end_date,
                end_date.year - 1,
            ),
        )

    if comparison == "previous_year_ytd":
        return (
            _safe_replace_year(
                start_date,
                start_date.year - 1,
            ),
            _safe_replace_year(
                end_date,
                end_date.year - 1,
            ),
        )

    if comparison == "custom_comparison":
        if (
            custom_start_date is None
            or custom_end_date is None
        ):
            return None

        return (
            min(
                custom_start_date,
                custom_end_date,
            ),
            max(
                custom_start_date,
                custom_end_date,
            ),
        )

    return None


def render_sidebar_filters(
    default_preset: str = "last_30_days",
    default_comparison: str = "previous_period",
) -> DashboardFilters:
    """Render shared language and date filters."""
    if "dashboard_language" not in st.session_state:
        st.session_state["dashboard_language"] = "tr"

    language_options = {
        "Türkçe": "tr",
        "English": "en",
    }

    current_language = st.session_state[
        "dashboard_language"
    ]

    selected_language_label = (
        st.sidebar.segmented_control(
            "Dil / Language",
            options=list(language_options.keys()),
            default=(
                "Türkçe"
                if current_language == "tr"
                else "English"
            ),
            key="dashboard_language_selector",
        )
    )

    language = language_options.get(
        selected_language_label,
        current_language,
    )

    st.session_state["dashboard_language"] = language

    st.sidebar.divider()

    st.sidebar.subheader(
        translate(
            "filters",
            language,
        )
    )

    st.sidebar.caption(
        translate(
            "filter_help",
            language,
        )
    )

    preset_labels = {
        translate(key, language): key
        for key in DATE_PRESET_KEYS
    }

    resolved_default_preset = (
        default_preset
        if default_preset in DATE_PRESET_KEYS
        else "last_30_days"
    )

    default_label = next(
        label
        for label, key in preset_labels.items()
        if key == resolved_default_preset
    )

    preset_options = list(
        preset_labels.keys()
    )

    selected_preset_label = st.sidebar.selectbox(
        translate(
            "date_range",
            language,
        ),
        options=preset_options,
        index=preset_options.index(
            default_label
        ),
        key="dashboard_date_preset",
    )

    preset = preset_labels[
        selected_preset_label
    ]

    default_start, default_end = (
        resolve_date_range(preset)
    )

    if preset == "custom_range":
        selected_dates = st.sidebar.date_input(
            translate(
                "date_range",
                language,
            ),
            value=(
                default_start,
                default_end,
            ),
            key="dashboard_custom_date_range",
        )

        if (
            isinstance(selected_dates, tuple)
            and len(selected_dates) == 2
        ):
            start_date, end_date = (
                selected_dates
            )
        else:
            start_date, end_date = (
                default_start,
                default_end,
            )

    else:
        start_date, end_date = (
            default_start,
            default_end,
        )

        st.sidebar.caption(
            f"{start_date:%d.%m.%Y} — "
            f"{end_date:%d.%m.%Y}"
        )

    normalized_start = min(
        start_date,
        end_date,
    )

    normalized_end = max(
        start_date,
        end_date,
    )

    start_date = normalized_start
    end_date = normalized_end

    comparison_labels = {
        translate(key, language): key
        for key in COMPARISON_KEYS
    }

    resolved_default_comparison = (
        default_comparison
        if default_comparison in COMPARISON_KEYS
        else "previous_period"
    )

    default_comparison_label = next(
        label
        for label, key in comparison_labels.items()
        if key == resolved_default_comparison
    )

    comparison_options = list(
        comparison_labels.keys()
    )

    selected_comparison_label = (
        st.sidebar.selectbox(
            translate(
                "comparison",
                language,
            ),
            options=comparison_options,
            index=comparison_options.index(
                default_comparison_label
            ),
            key="dashboard_comparison",
        )
    )

    comparison = comparison_labels[
        selected_comparison_label
    ]

    custom_comparison_start: date | None = None
    custom_comparison_end: date | None = None

    if comparison == "custom_comparison":
        default_comparison_range = (
            get_comparison_range(
                start_date=start_date,
                end_date=end_date,
                comparison="previous_period",
            )
        )

        if default_comparison_range is None:
            default_comparison_range = (
                start_date,
                end_date,
            )

        selected_comparison_dates = (
            st.sidebar.date_input(
                translate(
                    "custom_comparison",
                    language,
                ),
                value=default_comparison_range,
                key=(
                    "dashboard_custom_"
                    "comparison_range"
                ),
            )
        )

        if (
            isinstance(
                selected_comparison_dates,
                tuple,
            )
            and len(
                selected_comparison_dates
            ) == 2
        ):
            (
                custom_comparison_start,
                custom_comparison_end,
            ) = selected_comparison_dates

        else:
            (
                custom_comparison_start,
                custom_comparison_end,
            ) = default_comparison_range

    comparison_range = get_comparison_range(
        start_date=start_date,
        end_date=end_date,
        comparison=comparison,
        custom_start_date=(
            custom_comparison_start
        ),
        custom_end_date=(
            custom_comparison_end
        ),
    )

    same_length = True

    if comparison_range is not None:
        comparison_start = (
            comparison_range[0]
        )

        comparison_end = (
            comparison_range[1]
        )

        st.sidebar.caption(
            f"{translate('comparison', language)}: "
            f"{comparison_start:%d.%m.%Y} — "
            f"{comparison_end:%d.%m.%Y}"
        )

        same_length = (
            comparison_has_same_length(
                start_date=start_date,
                end_date=end_date,
                comparison_start_date=(
                    comparison_start
                ),
                comparison_end_date=(
                    comparison_end
                ),
            )
        )

        if not same_length:
            st.sidebar.warning(
                translate(
                    "comparison_length_warning",
                    language,
                )
            )

    st.sidebar.caption(
        translate(
            "selection_applies_to_all",
            language,
        )
    )

    return DashboardFilters(
        language=language,
        preset=preset,
        start_date=start_date,
        end_date=end_date,
        comparison=comparison,
        comparison_start_date=(
            comparison_range[0]
            if comparison_range is not None
            else None
        ),
        comparison_end_date=(
            comparison_range[1]
            if comparison_range is not None
            else None
        ),
        comparison_same_length=(
            same_length
        ),
    )


