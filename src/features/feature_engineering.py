from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd


SEASON_MAP = {
    12: ("Winter", "winter"),
    1: ("Winter", "winter"),
    2: ("Winter", "winter"),
    3: ("Spring", "spring"),
    4: ("Spring", "spring"),
    5: ("Spring", "spring"),
    6: ("Summer", "summer"),
    7: ("Summer", "summer"),
    8: ("Summer", "summer"),
    9: ("Autumn", "autumn"),
    10: ("Autumn", "autumn"),
    11: ("Autumn", "autumn"),
}

SEASON_ROAS_MULTIPLIER = {
    "winter": 1.15,
    "spring": 1.05,
    "summer": 0.95,
    "autumn": 1.10,
}

HOLIDAY_ROAS_MULTIPLIER = 1.20
PRE_HOLIDAY_DAYS = 7
POST_HOLIDAY_DAYS = 2


def _fallback_fixed_holidays(year: int) -> Dict[str, str]:
    """Return Turkey's fixed-date national holidays."""
    return {
        f"{year}-01-01": "Yılbaşı",
        f"{year}-04-23": "Ulusal Egemenlik ve Çocuk Bayramı",
        f"{year}-05-01": "Emek ve Dayanışma Günü",
        f"{year}-05-19": "Atatürk'ü Anma, Gençlik ve Spor Bayramı",
        f"{year}-07-15": "Demokrasi ve Millî Birlik Günü",
        f"{year}-08-30": "Zafer Bayramı",
        f"{year}-10-29": "Cumhuriyet Bayramı",
    }


def _fallback_religious_holidays() -> Dict[int, Dict[str, str]]:
    """
    Return fallback religious holiday dates.

    English names are intentionally stable because tests and downstream
    processing use these values independently from the installed language
    version of python-holidays.
    """
    return {
        2024: {
            "2024-04-09": "Eid al-Fitr Eve",
            "2024-04-10": "Eid al-Fitr",
            "2024-04-11": "Eid al-Fitr",
            "2024-04-12": "Eid al-Fitr",
            "2024-06-15": "Eid al-Adha Eve",
            "2024-06-16": "Eid al-Adha",
            "2024-06-17": "Eid al-Adha",
            "2024-06-18": "Eid al-Adha",
            "2024-06-19": "Eid al-Adha",
        },
        2025: {
            "2025-03-29": "Eid al-Fitr Eve",
            "2025-03-30": "Eid al-Fitr",
            "2025-03-31": "Eid al-Fitr",
            "2025-04-01": "Eid al-Fitr",
            "2025-06-05": "Eid al-Adha Eve",
            "2025-06-06": "Eid al-Adha",
            "2025-06-07": "Eid al-Adha",
            "2025-06-08": "Eid al-Adha",
            "2025-06-09": "Eid al-Adha",
        },
        2026: {
            "2026-03-19": "Eid al-Fitr Eve",
            "2026-03-20": "Eid al-Fitr",
            "2026-03-21": "Eid al-Fitr",
            "2026-03-22": "Eid al-Fitr",
            "2026-05-26": "Eid al-Adha Eve",
            "2026-05-27": "Eid al-Adha",
            "2026-05-28": "Eid al-Adha",
            "2026-05-29": "Eid al-Adha",
            "2026-05-30": "Eid al-Adha",
        },
    }


def get_turkey_public_holidays(year: int) -> Dict[str, str]:
    """Return Turkey public holidays for one year."""
    holiday_map: Dict[str, str] = {}

    try:
        import holidays

        tr_holidays = holidays.Turkey(years=[year])

        holiday_map.update(
            {
                pd.Timestamp(holiday_date).strftime("%Y-%m-%d"): str(name)
                for holiday_date, name in tr_holidays.items()
            }
        )

    except Exception:
        holiday_map.update(
            _fallback_fixed_holidays(year)
        )

    # Religious holiday names stay consistent across package versions.
    holiday_map.update(
        _fallback_religious_holidays().get(year, {})
    )

    return holiday_map


def build_holiday_map(
    date_from: str,
    date_to: str,
) -> Dict[str, str]:
    """Build a holiday map for every year in the analysis range."""
    if not str(date_from).strip() or not str(date_to).strip():
        raise ValueError(
            "date_from and date_to are required."
        )

    start = pd.to_datetime(
        date_from,
        errors="coerce",
    )

    end = pd.to_datetime(
        date_to,
        errors="coerce",
    )

    if pd.isna(start) or pd.isna(end):
        raise ValueError(
            "date_from and date_to must be valid dates."
        )

    if start > end:
        raise ValueError(
            "date_from cannot be later than date_to."
        )

    holiday_map: Dict[str, str] = {}

    for year in range(
        int(start.year),
        int(end.year) + 1,
    ):
        holiday_map.update(
            get_turkey_public_holidays(year)
        )

    return holiday_map


def _nearest_holiday_distance(
    current_date: date,
    holiday_dates: set[date],
    *,
    direction: str,
    limit: int,
) -> int:
    for offset in range(1, limit + 1):
        candidate = (
            current_date + timedelta(days=offset)
            if direction == "future"
            else current_date - timedelta(days=offset)
        )

        if candidate in holiday_dates:
            return offset

    return 0


def add_calendar_context_features(
    df: pd.DataFrame,
    holiday_map: Dict[str, str],
) -> pd.DataFrame:
    """Add season, holiday, pre-holiday and post-holiday features."""
    if "Date" not in df.columns:
        raise KeyError(
            "Date column is required."
        )

    result = df.copy()

    result["Date"] = pd.to_datetime(
        result["Date"],
        errors="coerce",
    )

    date_text = result["Date"].dt.strftime(
        "%Y-%m-%d"
    )

    result["IsHoliday"] = (
        date_text.isin(holiday_map).astype(int)
    )

    result["HolidayName"] = (
        date_text
        .map(holiday_map)
        .fillna("")
    )

    holiday_dates = {
        datetime.strptime(
            value,
            "%Y-%m-%d",
        ).date()
        for value in holiday_map
    }

    def context_values(
        value: pd.Timestamp,
    ) -> pd.Series:
        if pd.isna(value):
            return pd.Series(
                {
                    "DaysToHoliday": 0,
                    "DaysAfterHoliday": 0,
                    "IsPreHoliday": 0,
                    "IsPostHoliday": 0,
                }
            )

        current = value.date()
        is_holiday = current in holiday_dates

        days_to = (
            0
            if is_holiday
            else _nearest_holiday_distance(
                current,
                holiday_dates,
                direction="future",
                limit=PRE_HOLIDAY_DAYS,
            )
        )

        days_after = (
            0
            if is_holiday
            else _nearest_holiday_distance(
                current,
                holiday_dates,
                direction="past",
                limit=POST_HOLIDAY_DAYS,
            )
        )

        return pd.Series(
            {
                "DaysToHoliday": days_to,
                "DaysAfterHoliday": days_after,
                "IsPreHoliday": int(days_to > 0),
                "IsPostHoliday": int(days_after > 0),
            }
        )

    context = result["Date"].apply(
        context_values
    )

    for column in context.columns:
        result[column] = context[column].astype(int)

    result["Season"] = (
        result["Date"]
        .dt.month
        .map(
            lambda month: SEASON_MAP.get(
                month,
                ("Unknown", "unknown"),
            )[0]
        )
    )

    result["SeasonEN"] = (
        result["Date"]
        .dt.month
        .map(
            lambda month: SEASON_MAP.get(
                month,
                ("Unknown", "unknown"),
            )[1]
        )
    )

    result["SeasonROASMultiplier"] = (
        result["SeasonEN"]
        .map(SEASON_ROAS_MULTIPLIER)
        .fillna(1.0)
    )

    holiday_effect = np.select(
        [
            result["IsHoliday"].eq(1),
            result["IsPreHoliday"].eq(1),
            result["IsPostHoliday"].eq(1),
        ],
        [
            HOLIDAY_ROAS_MULTIPLIER,
            1.20,
            0.95,
        ],
        default=1.0,
    )

    result["ExpectedROASMultiplier"] = (
        result["SeasonROASMultiplier"]
        * holiday_effect
    )

    return result


def compute_kpis(
    df: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = [
        "Spend",
        "Impressions",
        "Clicks",
        "Conversions",
        "ConversionValue",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing required KPI columns: "
            f"{', '.join(missing_columns)}"
        )

    result = df.copy()

    for column in required_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).fillna(0.0)

    result["CTR"] = (
        result["Clicks"]
        / result["Impressions"].replace(0, np.nan)
    )

    result["CPC"] = (
        result["Spend"]
        / result["Clicks"].replace(0, np.nan)
    )

    result["ConvRate"] = (
        result["Conversions"]
        / result["Clicks"].replace(0, np.nan)
    )

    result["CPA"] = (
        result["Spend"]
        / result["Conversions"].replace(0, np.nan)
    )

    result["ROAS"] = (
        result["ConversionValue"]
        / result["Spend"].replace(0, np.nan)
    )

    result["Profit"] = (
        result["ConversionValue"]
        - result["Spend"]
    )

    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns

    result[numeric_columns] = (
        result[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    return result


def compute_roas_target_gap(
    df: pd.DataFrame,
    target_roas: float,
) -> pd.DataFrame:
    if "ROAS" not in df.columns:
        raise KeyError(
            "ROAS column is required."
        )

    result = df.copy()

    result["TargetROAS"] = float(
        target_roas
    )

    result["ROASGap"] = (
        result["ROAS"]
        - float(target_roas)
    )

    result["ROASGapPct"] = np.where(
        target_roas > 0,
        (
            result["ROASGap"]
            / float(target_roas)
        ) * 100,
        0.0,
    )

    result["ROASStatus"] = np.select(
        [
            result["ROAS"] >= target_roas * 1.10,
            result["ROAS"] >= target_roas * 0.90,
        ],
        [
            "Above Target",
            "On Target",
        ],
        default="Below Target",
    )

    return result


def add_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if "Date" not in df.columns:
        raise KeyError(
            "Date column is required."
        )

    result = df.copy()

    result["Date"] = pd.to_datetime(
        result["Date"],
        errors="coerce",
    )

    result["DayOfWeek"] = (
        result["Date"]
        .dt.dayofweek
        .fillna(0)
        .astype(int)
    )

    result["DayOfMonth"] = (
        result["Date"]
        .dt.day
        .fillna(0)
        .astype(int)
    )

    result["MonthNum"] = (
        result["Date"]
        .dt.month
        .fillna(0)
        .astype(int)
    )

    result["WeekOfYear"] = (
        result["Date"]
        .dt.isocalendar()
        .week
        .fillna(0)
        .astype(int)
    )

    result["IsWeekend"] = (
        result["DayOfWeek"] >= 5
    ).astype(int)

    result["Quarter"] = (
        result["Date"]
        .dt.quarter
        .fillna(0)
        .astype(int)
    )

    result["Year"] = (
        result["Date"]
        .dt.year
        .fillna(0)
        .astype(int)
    )

    return result


def add_lag_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = [
        "CampaignId",
        "Date",
        "Spend",
        "Clicks",
        "Conversions",
        "ConversionValue",
        "ROAS",
        "CPA",
        "CTR",
        "ConvRate",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing required lag columns: "
            f"{', '.join(missing_columns)}"
        )

    result = df.copy()

    result["Date"] = pd.to_datetime(
        result["Date"],
        errors="coerce",
    )

    result = (
        result
        .sort_values(
            ["CampaignId", "Date"]
        )
        .reset_index(drop=True)
    )

    grouped = result.groupby(
        "CampaignId",
        sort=False,
    )

    lag_columns = [
        "Spend",
        "Clicks",
        "Conversions",
        "ConversionValue",
        "ROAS",
        "CPA",
        "CTR",
        "ConvRate",
    ]

    for column in lag_columns:
        result[f"{column}_lag_1"] = (
            grouped[column].shift(1)
        )

    # Hedef bir sonraki gün olduğu için mevcut günün değeri,
    # son yedi günlük ortalamanın içine güvenle girebilir.
    rolling_columns = [
        "Spend",
        "Clicks",
        "Conversions",
        "ConversionValue",
    ]

    for column in rolling_columns:
        result[f"{column}_lag_7_avg"] = (
            grouped[column].transform(
                lambda series: (
                    series
                    .rolling(
                        window=7,
                        min_periods=1,
                    )
                    .mean()
                )
            )
        )

    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns

    result[numeric_columns] = (
        result[numeric_columns]
        .fillna(0.0)
    )

    return result


def prepare_feature_data(
    ads_raw: pd.DataFrame,
    holiday_map: Dict[str, str],
) -> pd.DataFrame:
    """Create model features without removing the latest campaign day."""
    if ads_raw.empty:
        return ads_raw.copy()

    result = compute_kpis(
        ads_raw
    )

    result = add_time_features(
        result
    )

    result = add_calendar_context_features(
        result,
        holiday_map,
    )

    result = add_lag_features(
        result
    )

    return (
        result
        .sort_values(
            ["CampaignId", "Date"]
        )
        .reset_index(drop=True)
    )


def prepare_training_data(
    ads_raw: pd.DataFrame,
    holiday_map: Dict[str, str],
) -> pd.DataFrame:
    """Create supervised next-day targets from leakage-safe features."""
    result = prepare_feature_data(
        ads_raw,
        holiday_map,
    )

    if result.empty:
        return result

    grouped = result.groupby(
        "CampaignId",
        sort=False,
    )

    result["Target_Conversions_Next"] = (
        grouped["Conversions"].shift(-1)
    )

    result["Target_Revenue_Next"] = (
        grouped["ConversionValue"].shift(-1)
    )

    return result.dropna(
        subset=[
            "Target_Conversions_Next",
            "Target_Revenue_Next",
        ]
    ).copy()


def get_feature_columns() -> List[str]:
    """
    Return predictive ML features.

    Identifier columns such as CampaignId are intentionally excluded
    from model training to prevent identifier leakage and memorization.
    CampaignId remains available elsewhere for grouping, lag features,
    joins, reporting, and campaign-level explanations.
    """
    return [
        "Spend",
        "Impressions",
        "Clicks",
        "Conversions",
        "ConversionValue",
        "CTR",
        "CPC",
        "ConvRate",
        "CPA",
        "ROAS",
        "Profit",
        "DayOfWeek",
        "DayOfMonth",
        "MonthNum",
        "WeekOfYear",
        "IsWeekend",
        "Quarter",
        "Year",
        "IsHoliday",
        "IsPreHoliday",
        "IsPostHoliday",
        "DaysToHoliday",
        "DaysAfterHoliday",
        "SeasonROASMultiplier",
        "ExpectedROASMultiplier",
        "Spend_lag_1",
        "Spend_lag_7_avg",
        "Clicks_lag_1",
        "Clicks_lag_7_avg",
        "Conversions_lag_1",
        "Conversions_lag_7_avg",
        "ConversionValue_lag_1",
        "ConversionValue_lag_7_avg",
        "ROAS_lag_1",
        "CPA_lag_1",
        "CTR_lag_1",
        "ConvRate_lag_1",
    ]


def get_latest_campaign_state(
    ads_raw: pd.DataFrame,
    holiday_map: Dict[str, str],
) -> pd.DataFrame:
    """Return the latest model-eligible row for each campaign."""
    result = prepare_training_data(
        ads_raw=ads_raw,
        holiday_map=holiday_map,
    )

    if result.empty:
        return result

    group_columns = [
        column
        for column in [
            "CampaignId",
            "Campaign",
            "Channel",
        ]
        if column in result.columns
    ]

    if "CampaignId" not in group_columns:
        raise KeyError(
            "CampaignId column is required."
        )

    return (
        result
        .sort_values(
            ["CampaignId", "Date"]
        )
        .groupby(
            group_columns,
            as_index=False,
            sort=False,
        )
        .tail(1)
        .reset_index(drop=True)
    )