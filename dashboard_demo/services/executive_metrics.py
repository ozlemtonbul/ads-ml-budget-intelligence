from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from dashboard_demo.services.executive_data import (
    DateCoverage,
)
from dashboard_demo.utils import (
    find_first_column,
    safe_sum,
)


MINIMUM_COMPARISON_COVERAGE = 0.80


@dataclass(frozen=True)
class ExecutiveKPIs:
    """Core executive advertising KPIs."""

    spend: float
    revenue: float
    conversions: float
    profit: float
    clicks: float
    impressions: float
    roas: float
    cpa: float
    conversion_rate: float
    ctr: float


@dataclass(frozen=True)
class KPIComparison:
    """
    KPI comparison results with coverage validation.
    """

    is_available: bool
    coverage_ratio: float
    revenue_change_pct: Optional[float]
    spend_change_pct: Optional[float]
    conversions_change_pct: Optional[float]
    profit_change_pct: Optional[float]
    roas_change_pct: Optional[float]
    cpa_change_pct: Optional[float]


def calculate_kpis(
    dataframe: pd.DataFrame,
) -> ExecutiveKPIs:
    """
    Calculate weighted executive KPIs.

    ROAS, CPA, conversion rate and CTR are calculated
    from their component totals. Row-level ratios are
    not averaged.
    """

    spend = safe_sum(
        dataframe,
        ["Spend", "Cost", "AdSpend"],
    )

    revenue = safe_sum(
        dataframe,
        [
            "ConversionValue",
            "Revenue",
            "PurchaseRevenue",
        ],
    )

    conversions = safe_sum(
        dataframe,
        [
            "Conversions",
            "Purchases",
            "Transactions",
        ],
    )

    profit = safe_sum(
        dataframe,
        ["Profit", "PredictedProfit"],
    )

    clicks = safe_sum(
        dataframe,
        ["Clicks"],
    )

    impressions = safe_sum(
        dataframe,
        ["Impressions"],
    )

    roas = (
        revenue / spend
        if spend > 0
        else 0.0
    )

    cpa = (
        spend / conversions
        if conversions > 0
        else 0.0
    )

    conversion_rate = (
        conversions / clicks
        if clicks > 0
        else 0.0
    )

    ctr = (
        clicks / impressions
        if impressions > 0
        else 0.0
    )

    return ExecutiveKPIs(
        spend=spend,
        revenue=revenue,
        conversions=conversions,
        profit=profit,
        clicks=clicks,
        impressions=impressions,
        roas=roas,
        cpa=cpa,
        conversion_rate=conversion_rate,
        ctr=ctr,
    )


def percentage_change(
    current_value: float,
    previous_value: float,
) -> Optional[float]:
    """
    Calculate percentage change safely.
    """

    if previous_value == 0:
        return None

    return (
        (current_value - previous_value)
        / abs(previous_value)
    ) * 100


def build_kpi_comparison(
    current: ExecutiveKPIs,
    previous: ExecutiveKPIs,
    comparison_coverage: DateCoverage,
    minimum_coverage: float = (
        MINIMUM_COMPARISON_COVERAGE
    ),
) -> KPIComparison:
    """
    Calculate KPI changes only when comparison coverage
    is sufficient.

    Default minimum comparison coverage is 80%.
    """

    comparison_is_available = (
        comparison_coverage.is_sufficient(
            minimum_ratio=minimum_coverage
        )
    )

    if not comparison_is_available:
        return KPIComparison(
            is_available=False,
            coverage_ratio=(
                comparison_coverage.coverage_ratio
            ),
            revenue_change_pct=None,
            spend_change_pct=None,
            conversions_change_pct=None,
            profit_change_pct=None,
            roas_change_pct=None,
            cpa_change_pct=None,
        )

    return KPIComparison(
        is_available=True,
        coverage_ratio=(
            comparison_coverage.coverage_ratio
        ),
        revenue_change_pct=percentage_change(
            current.revenue,
            previous.revenue,
        ),
        spend_change_pct=percentage_change(
            current.spend,
            previous.spend,
        ),
        conversions_change_pct=percentage_change(
            current.conversions,
            previous.conversions,
        ),
        profit_change_pct=percentage_change(
            current.profit,
            previous.profit,
        ),
        roas_change_pct=percentage_change(
            current.roas,
            previous.roas,
        ),
        cpa_change_pct=percentage_change(
            current.cpa,
            previous.cpa,
        ),
    )


def _numeric_series(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> pd.Series:
    """
    Return a numeric float series for the first
    matching column.
    """

    column = find_first_column(
        dataframe,
        candidates,
    )

    if column is None:
        return pd.Series(
            0.0,
            index=dataframe.index,
            dtype="float64",
        )

    return (
        pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )
        .fillna(0.0)
        .astype("float64")
    )


def build_daily_trend(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate daily performance and calculate
    weighted ratios and moving averages.
    """

    if dataframe.empty:
        return pd.DataFrame()

    date_column = find_first_column(
        dataframe,
        ["Date", "Day"],
    )

    if date_column is None:
        return pd.DataFrame()

    working = pd.DataFrame(
        index=dataframe.index,
    )

    working["Date"] = pd.to_datetime(
        dataframe[date_column],
        errors="coerce",
    )

    working["Spend"] = _numeric_series(
        dataframe,
        ["Spend", "Cost", "AdSpend"],
    )

    working["Revenue"] = _numeric_series(
        dataframe,
        [
            "ConversionValue",
            "Revenue",
            "PurchaseRevenue",
        ],
    )

    working["Conversions"] = _numeric_series(
        dataframe,
        [
            "Conversions",
            "Purchases",
            "Transactions",
        ],
    )

    working["Profit"] = _numeric_series(
        dataframe,
        ["Profit", "PredictedProfit"],
    )

    working["Clicks"] = _numeric_series(
        dataframe,
        ["Clicks"],
    )

    working["Impressions"] = _numeric_series(
        dataframe,
        ["Impressions"],
    )

    working = working.dropna(
        subset=["Date"]
    )

    if working.empty:
        return pd.DataFrame()

    daily = (
        working.groupby(
            "Date",
            as_index=False,
        )
        .agg(
            Spend=("Spend", "sum"),
            Revenue=("Revenue", "sum"),
            Conversions=("Conversions", "sum"),
            Profit=("Profit", "sum"),
            Clicks=("Clicks", "sum"),
            Impressions=("Impressions", "sum"),
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    daily["ROAS"] = (
        daily["Revenue"]
        .div(
            daily["Spend"].where(
                daily["Spend"] > 0
            )
        )
        .fillna(0.0)
    )

    daily["CPA"] = (
        daily["Spend"]
        .div(
            daily["Conversions"].where(
                daily["Conversions"] > 0
            )
        )
        .fillna(0.0)
    )

    daily["ConversionRate"] = (
        daily["Conversions"]
        .div(
            daily["Clicks"].where(
                daily["Clicks"] > 0
            )
        )
        .fillna(0.0)
    )

    daily["CTR"] = (
        daily["Clicks"]
        .div(
            daily["Impressions"].where(
                daily["Impressions"] > 0
            )
        )
        .fillna(0.0)
    )

    daily["RevenueMA7"] = (
        daily["Revenue"]
        .rolling(
            window=7,
            min_periods=1,
        )
        .mean()
    )

    daily["RevenueMA30"] = (
        daily["Revenue"]
        .rolling(
            window=30,
            min_periods=1,
        )
        .mean()
    )

    daily["SpendMA7"] = (
        daily["Spend"]
        .rolling(
            window=7,
            min_periods=1,
        )
        .mean()
    )

    daily["SpendMA30"] = (
        daily["Spend"]
        .rolling(
            window=30,
            min_periods=1,
        )
        .mean()
    )

    rolling_revenue_7 = (
        daily["Revenue"]
        .rolling(
            window=7,
            min_periods=1,
        )
        .sum()
    )

    rolling_spend_7 = (
        daily["Spend"]
        .rolling(
            window=7,
            min_periods=1,
        )
        .sum()
    )

    rolling_revenue_30 = (
        daily["Revenue"]
        .rolling(
            window=30,
            min_periods=1,
        )
        .sum()
    )

    rolling_spend_30 = (
        daily["Spend"]
        .rolling(
            window=30,
            min_periods=1,
        )
        .sum()
    )

    daily["ROASMA7"] = (
        rolling_revenue_7
        .div(
            rolling_spend_7.where(
                rolling_spend_7 > 0
            )
        )
        .fillna(0.0)
    )

    daily["ROASMA30"] = (
        rolling_revenue_30
        .div(
            rolling_spend_30.where(
                rolling_spend_30 > 0
            )
        )
        .fillna(0.0)
    )

    return daily


def calculate_model_r2(
    model_metrics: pd.DataFrame,
) -> Optional[float]:
    """
    Return average validation R-squared as a percentage.
    """

    if (
        model_metrics.empty
        or "R2" not in model_metrics.columns
    ):
        return None

    values = pd.to_numeric(
        model_metrics["R2"],
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    normalized_values = values.clip(
        lower=0.0,
        upper=1.0,
    )

    return float(
        normalized_values.mean() * 100
    )


def format_delta(
    value: Optional[float],
) -> Optional[str]:
    """
    Format a Streamlit-compatible delta value.
    """

    if value is None:
        return None

    return f"{value:+.1f}%"


