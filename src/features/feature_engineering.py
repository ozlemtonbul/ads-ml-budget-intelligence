from datetime import datetime, timedelta
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


def get_turkey_public_holidays(year: int) -> Dict[str, str]:
    holiday_map: Dict[str, str] = {}

    try:
        import holidays

        tr_holidays = holidays.Turkey(years=year)
        holiday_map.update({str(date): name for date, name in tr_holidays.items()})

    except Exception:
        holiday_map.update(
            {
                f"{year}-01-01": "New Year",
                f"{year}-04-23": "National Sovereignty and Children Day",
                f"{year}-05-01": "Labour Day",
                f"{year}-05-19": "Commemoration of Ataturk, Youth and Sports Day",
                f"{year}-07-15": "Democracy and National Unity Day",
                f"{year}-08-30": "Victory Day",
                f"{year}-10-29": "Republic Day",
            }
        )

    religious_holidays = {
        2024: {
            "2024-04-09": "Eid al-Fitr Eve",
            "2024-04-10": "Eid al-Fitr",
            "2024-04-11": "Eid al-Fitr",
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
            "2026-05-26": "Eid al-Adha Eve",
            "2026-05-27": "Eid al-Adha",
            "2026-05-28": "Eid al-Adha",
            "2026-05-29": "Eid al-Adha",
            "2026-05-30": "Eid al-Adha",
        },
    }

    holiday_map.update(religious_holidays.get(year, {}))

    return holiday_map


def build_holiday_map(date_from: str, date_to: str) -> Dict[str, str]:
    if not date_from or not date_to:
        raise ValueError("date_from and date_to are required.")

    start_year = int(date_from[:4])
    end_year = int(date_to[:4])

    if start_year > end_year:
        raise ValueError("date_from cannot be later than date_to.")

    holiday_map: Dict[str, str] = {}

    for year in range(start_year, end_year + 1):
        holiday_map.update(get_turkey_public_holidays(year))

    return holiday_map


def add_calendar_context_features(
    df: pd.DataFrame,
    holiday_map: Dict[str, str],
) -> pd.DataFrame:
    df = df.copy()

    if "Date" not in df.columns:
        raise KeyError("Date column is required.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    date_str = df["Date"].dt.strftime("%Y-%m-%d")

    df["IsHoliday"] = date_str.map(
        lambda value: int(pd.notna(value) and value in holiday_map)
    )

    df["HolidayName"] = date_str.map(
        lambda value: holiday_map.get(value, "") if pd.notna(value) else ""
    )

    holiday_dates = {
        datetime.strptime(date_value, "%Y-%m-%d").date()
        for date_value in holiday_map
    }

    def is_pre_holiday(value) -> int:
        if pd.isna(value):
            return 0

        current_date = value.date()

        return int(
            any(
                current_date + timedelta(days=day_offset) in holiday_dates
                for day_offset in range(1, 4)
            )
        )

    df["IsPreHoliday"] = df["Date"].map(is_pre_holiday)

    df["Season"] = df["Date"].dt.month.map(
        lambda month: SEASON_MAP.get(month, ("Unknown", "unknown"))[0]
    )

    df["SeasonEN"] = df["Date"].dt.month.map(
        lambda month: SEASON_MAP.get(month, ("Unknown", "unknown"))[1]
    )

    df["SeasonROASMultiplier"] = (
        df["SeasonEN"]
        .map(SEASON_ROAS_MULTIPLIER)
        .fillna(1.0)
    )

    df["ExpectedROASMultiplier"] = df.apply(
        lambda row: (
            HOLIDAY_ROAS_MULTIPLIER * row["SeasonROASMultiplier"]
            if row["IsHoliday"] or row["IsPreHoliday"]
            else row["SeasonROASMultiplier"]
        ),
        axis=1,
    )

    return df


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "Spend",
        "Impressions",
        "Clicks",
        "Conversions",
        "ConversionValue",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required KPI columns: {', '.join(missing_columns)}"
        )

    df = df.copy()

    df["CTR"] = (
        df["Clicks"]
        / df["Impressions"].replace(0, np.nan)
    )

    df["CPC"] = (
        df["Spend"]
        / df["Clicks"].replace(0, np.nan)
    )

    df["ConvRate"] = (
        df["Conversions"]
        / df["Clicks"].replace(0, np.nan)
    )

    df["CPA"] = (
        df["Spend"]
        / df["Conversions"].replace(0, np.nan)
    )

    df["ROAS"] = (
        df["ConversionValue"]
        / df["Spend"].replace(0, np.nan)
    )

    df["Profit"] = (
        df["ConversionValue"]
        - df["Spend"]
    )

    return (
        df.replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )


def compute_roas_target_gap(
    df: pd.DataFrame,
    target_roas: float,
) -> pd.DataFrame:
    if "ROAS" not in df.columns:
        raise KeyError("ROAS column is required.")

    df = df.copy()

    df["TargetROAS"] = target_roas
    df["ROASGap"] = df["ROAS"] - target_roas

    df["ROASGapPct"] = np.where(
        target_roas > 0,
        (df["ROASGap"] / target_roas) * 100,
        0,
    )

    df["ROASStatus"] = df["ROAS"].apply(
        lambda roas: (
            "Above Target"
            if roas >= target_roas * 1.10
            else (
                "On Target"
                if roas >= target_roas * 0.90
                else "Below Target"
            )
        )
    )

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        raise KeyError("Date column is required.")

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df["DayOfWeek"] = df["Date"].dt.dayofweek.fillna(0).astype(int)
    df["DayOfMonth"] = df["Date"].dt.day.fillna(0).astype(int)
    df["MonthNum"] = df["Date"].dt.month.fillna(0).astype(int)
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.fillna(0).astype(int)
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["Quarter"] = df["Date"].dt.quarter.fillna(0).astype(int)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
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
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required lag columns: {', '.join(missing_columns)}"
        )

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["CampaignId", "Date"])

    grouped = df.groupby("CampaignId")

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
        df[f"{column}_lag_1"] = grouped[column].shift(1)

    rolling_average_columns = [
        "Spend",
        "Clicks",
        "Conversions",
        "ConversionValue",
    ]

    for column in rolling_average_columns:
        df[f"{column}_lag_7_avg"] = (
            grouped[column]
            .rolling(window=7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return df.fillna(0)


def prepare_training_data(
    ads_raw: pd.DataFrame,
    holiday_map: Dict[str, str],
) -> pd.DataFrame:
    if ads_raw.empty:
        return ads_raw.copy()

    df = ads_raw.copy()

    df = compute_kpis(df)
    df = add_time_features(df)
    df = add_calendar_context_features(df, holiday_map)
    df = add_lag_features(df)

    df = df.sort_values(["CampaignId", "Date"]).copy()

    grouped = df.groupby("CampaignId")

    df["Target_Conversions_Next"] = (
        grouped["Conversions"].shift(-1)
    )

    df["Target_Revenue_Next"] = (
        grouped["ConversionValue"].shift(-1)
    )

    return df.dropna(
        subset=[
            "Target_Conversions_Next",
            "Target_Revenue_Next",
        ]
    ).copy()


def get_feature_columns() -> List[str]:
    return [
        "CampaignId",
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
        "IsHoliday",
        "IsPreHoliday",
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
    df = prepare_training_data(
        ads_raw=ads_raw,
        holiday_map=holiday_map,
    )

    if df.empty:
        return df

    latest_df = (
        df.sort_values(["CampaignId", "Date"])
        .groupby(
            ["CampaignId", "Campaign", "Channel"],
            as_index=False,
        )
        .tail(1)
        .reset_index(drop=True)
    )

    return latest_df