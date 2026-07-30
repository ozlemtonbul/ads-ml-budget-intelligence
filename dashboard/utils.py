from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from dashboard.app_config import OUTPUT_DIR


@st.cache_data(ttl=300)
def load_csv(file_name: str) -> pd.DataFrame:
    file_path = OUTPUT_DIR / file_name

    if not file_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(file_path)
    except Exception as exc:
        st.warning(f"Could not load {file_name}: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_text(file_name: str) -> str:
    file_path = OUTPUT_DIR / file_name

    if not file_path.exists():
        return ""

    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as exc:
        st.warning(f"Could not load {file_name}: {exc}")
        return ""


def find_first_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> Optional[str]:
    if dataframe.empty:
        return None

    normalized_columns = {
        column.lower().replace("_", "").replace(" ", ""): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalized_candidate = (
            candidate.lower()
            .replace("_", "")
            .replace(" ", "")
        )

        if normalized_candidate in normalized_columns:
            return normalized_columns[normalized_candidate]

    return None


def safe_sum(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> float:
    column = find_first_column(dataframe, candidates)

    if column is None:
        return 0.0

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )

    return float(values.fillna(0).sum())


def safe_mean(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> float:
    column = find_first_column(dataframe, candidates)

    if column is None:
        return 0.0

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return 0.0

    return float(values.mean())


def format_currency(value: float) -> str:
    return f"₺{value:,.0f}"


def format_number(value: float) -> str:
    return f"{value:,.0f}"


def format_ratio(value: float) -> str:
    return f"{value:,.2f}x"


def get_latest_output_time() -> str:
    if not OUTPUT_DIR.exists():
        return "No output found"

    files = list(OUTPUT_DIR.glob("*"))

    if not files:
        return "No output found"

    latest_file = max(
        files,
        key=lambda path: path.stat().st_mtime,
    )

    modified_time = pd.Timestamp(
        latest_file.stat().st_mtime,
        unit="s",
    )

    return modified_time.strftime("%Y-%m-%d %H:%M")
