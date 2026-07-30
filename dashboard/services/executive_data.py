from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from dashboard.utils import (
    find_first_column,
    load_csv,
    load_text,
)


@dataclass(frozen=True)
class ExecutiveDataBundle:
    """All source data required by Executive Overview."""

    daily: pd.DataFrame
    recommendations: pd.DataFrame
    portfolio: pd.DataFrame
    recommendation_summary: pd.DataFrame
    model_metrics: pd.DataFrame
    executive_commentary: str


@dataclass(frozen=True)
class DateCoverage:
    """Data availability information for a date range."""

    selected_days: int
    available_days: int
    coverage_ratio: float
    available_start: Optional[date]
    available_end: Optional[date]

    @property
    def has_data(self) -> bool:
        return self.available_days > 0

    @property
    def is_complete(self) -> bool:
        return self.coverage_ratio >= 1.0

    def is_sufficient(
        self,
        minimum_ratio: float = 0.80,
    ) -> bool:
        return (
            self.available_days > 0
            and self.coverage_ratio >= minimum_ratio
        )


def load_executive_data() -> ExecutiveDataBundle:
    """
    Load all Executive Overview output files.
    """

    return ExecutiveDataBundle(
        daily=load_csv(
            "ads_daily_fact.csv"
        ),
        recommendations=load_csv(
            "ads_budget_optimization_recommendations.csv"
        ),
        portfolio=load_csv(
            "ads_portfolio_budget_allocation.csv"
        ),
        recommendation_summary=load_csv(
            "ads_recommendation_summary.csv"
        ),
        model_metrics=load_csv(
            "ads_model_validation_metrics.csv"
        ),
        executive_commentary=load_text(
            "ads_portfolio_executive_commentary.txt"
        ),
    )


def get_date_column(
    dataframe: pd.DataFrame,
) -> Optional[str]:
    """
    Find the daily date column.
    """

    return find_first_column(
        dataframe,
        ["Date", "Day"],
    )


def get_available_date_bounds(
    dataframe: pd.DataFrame,
) -> tuple[Optional[date], Optional[date]]:
    """
    Return the first and last valid dates in a dataframe.
    """

    if dataframe.empty:
        return None, None

    date_column = get_date_column(dataframe)

    if date_column is None:
        return None, None

    parsed_dates = pd.to_datetime(
        dataframe[date_column],
        errors="coerce",
    ).dropna()

    if parsed_dates.empty:
        return None, None

    return (
        parsed_dates.min().date(),
        parsed_dates.max().date(),
    )


def filter_by_date(
    dataframe: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Filter rows using an inclusive date range.
    """

    if dataframe.empty:
        return dataframe.copy()

    date_column = get_date_column(dataframe)

    if date_column is None:
        return dataframe.copy()

    normalized_start = min(
        start_date,
        end_date,
    )

    normalized_end = max(
        start_date,
        end_date,
    )

    result = dataframe.copy()

    result[date_column] = pd.to_datetime(
        result[date_column],
        errors="coerce",
    )

    valid_date_mask = (
        result[date_column]
        .dt.date
        .between(
            normalized_start,
            normalized_end,
        )
    )

    return (
        result.loc[valid_date_mask]
        .copy()
        .reset_index(drop=True)
    )


def count_unique_data_days(
    dataframe: pd.DataFrame,
) -> int:
    """
    Count unique valid dates in a dataframe.
    """

    if dataframe.empty:
        return 0

    date_column = get_date_column(dataframe)

    if date_column is None:
        return 0

    parsed_dates = pd.to_datetime(
        dataframe[date_column],
        errors="coerce",
    ).dropna()

    return int(
        parsed_dates.dt.date.nunique()
    )


def calculate_date_coverage(
    source_dataframe: pd.DataFrame,
    filtered_dataframe: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> DateCoverage:
    """
    Calculate selected days, available days and coverage.
    """

    normalized_start = min(
        start_date,
        end_date,
    )

    normalized_end = max(
        start_date,
        end_date,
    )

    selected_days = (
        normalized_end - normalized_start
    ).days + 1

    available_days = count_unique_data_days(
        filtered_dataframe
    )

    coverage_ratio = (
        available_days / selected_days
        if selected_days > 0
        else 0.0
    )

    coverage_ratio = min(
        max(coverage_ratio, 0.0),
        1.0,
    )

    available_start, available_end = (
        get_available_date_bounds(
            source_dataframe
        )
    )

    return DateCoverage(
        selected_days=selected_days,
        available_days=available_days,
        coverage_ratio=coverage_ratio,
        available_start=available_start,
        available_end=available_end,
    )


def calculate_data_age_days(
    available_end: Optional[date],
    today: Optional[date] = None,
) -> Optional[int]:
    """
    Return how many days old the latest data is.
    """

    if available_end is None:
        return None

    reference_date = today or date.today()

    return max(
        (reference_date - available_end).days,
        0,
    )


def recommendation_period_is_known(
    recommendations: pd.DataFrame,
) -> bool:
    """
    Return whether recommendation output contains
    explicit analysis-period metadata.
    """

    required_columns = {
        "AnalysisStartDate",
        "AnalysisEndDate",
    }

    return required_columns.issubset(
        recommendations.columns
    )


def get_recommendation_period(
    recommendations: pd.DataFrame,
) -> tuple[Optional[date], Optional[date]]:
    """
    Read recommendation analysis dates when available.
    """

    if not recommendation_period_is_known(
        recommendations
    ):
        return None, None

    start_values = pd.to_datetime(
        recommendations["AnalysisStartDate"],
        errors="coerce",
    ).dropna()

    end_values = pd.to_datetime(
        recommendations["AnalysisEndDate"],
        errors="coerce",
    ).dropna()

    if start_values.empty or end_values.empty:
        return None, None

    return (
        start_values.min().date(),
        end_values.max().date(),
    )