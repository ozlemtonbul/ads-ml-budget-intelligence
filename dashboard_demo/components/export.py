from __future__ import annotations

import re
from io import BytesIO
from typing import Mapping

import pandas as pd
import streamlit as st


EXCEL_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument."
    "spreadsheetml.sheet"
)


def normalize_file_name(value: str) -> str:
    """
    Return a safe file name without an extension.
    """

    normalized = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        value.strip(),
    )

    normalized = normalized.strip("_")

    return normalized or "dashboard_export"


def normalize_sheet_name(
    value: str,
    used_names: set[str],
) -> str:
    """
    Return a valid and unique Excel worksheet name.
    """

    cleaned = re.sub(
        r"[\[\]:*?/\\]",
        "_",
        value.strip(),
    )

    cleaned = cleaned[:31] or "Data"
    candidate = cleaned
    counter = 1

    while candidate in used_names:
        suffix = f"_{counter}"
        candidate = (
            cleaned[: 31 - len(suffix)]
            + suffix
        )
        counter += 1

    used_names.add(candidate)

    return candidate


def dataframe_to_csv_bytes(
    dataframe: pd.DataFrame,
) -> bytes:
    """
    Convert a dataframe to an Excel-compatible UTF-8 CSV.
    """

    if dataframe is None:
        dataframe = pd.DataFrame()

    csv_text = dataframe.to_csv(
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )

    return csv_text.encode("utf-8-sig")


def dataframes_to_excel_bytes(
    sheets: Mapping[str, pd.DataFrame],
) -> bytes:
    """
    Create one Excel workbook containing multiple sheets.
    """

    buffer = BytesIO()
    used_sheet_names: set[str] = set()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl",
    ) as writer:
        valid_sheet_count = 0

        for requested_name, dataframe in sheets.items():
            if dataframe is None:
                continue

            sheet_name = normalize_sheet_name(
                requested_name,
                used_sheet_names,
            )

            dataframe.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

            worksheet = writer.book[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = (
                worksheet.dimensions
            )

            for column_cells in worksheet.columns:
                maximum_length = 0
                column_letter = (
                    column_cells[0].column_letter
                )

                for cell in column_cells:
                    cell_value = (
                        ""
                        if cell.value is None
                        else str(cell.value)
                    )

                    maximum_length = max(
                        maximum_length,
                        len(cell_value),
                    )

                worksheet.column_dimensions[
                    column_letter
                ].width = min(
                    max(maximum_length + 2, 10),
                    45,
                )

            valid_sheet_count += 1

        if valid_sheet_count == 0:
            pd.DataFrame(
                {
                    "Message": [
                        "No data is available for export."
                    ]
                }
            ).to_excel(
                writer,
                sheet_name="Data",
                index=False,
            )

    buffer.seek(0)

    return buffer.getvalue()


def render_export_buttons(
    csv_dataframe: pd.DataFrame,
    excel_sheets: Mapping[str, pd.DataFrame],
    file_name: str,
    language: str,
    key_prefix: str,
) -> None:
    """
    Render shared CSV and Excel download buttons.

    CSV contains the principal visible table.
    Excel contains all supplied worksheets.
    """

    safe_file_name = normalize_file_name(file_name)

    csv_label = (
        "CSV İndir"
        if language == "tr"
        else "Download CSV"
    )

    excel_label = (
        "Excel İndir"
        if language == "tr"
        else "Download Excel"
    )

    no_data_message = (
        "İndirilecek veri bulunmuyor."
        if language == "tr"
        else "No data is available for download."
    )

    csv_is_empty = (
        csv_dataframe is None
        or csv_dataframe.empty
    )

    excel_has_data = any(
        dataframe is not None
        and not dataframe.empty
        for dataframe in excel_sheets.values()
    )

    csv_column, excel_column = st.columns(2)

    with csv_column:
        st.download_button(
            label=csv_label,
            data=(
                dataframe_to_csv_bytes(
                    csv_dataframe
                )
                if not csv_is_empty
                else b""
            ),
            file_name=f"{safe_file_name}.csv",
            mime="text/csv; charset=utf-8",
            disabled=csv_is_empty,
            key=f"{key_prefix}_csv_download",
            width="stretch",
        )

    with excel_column:
        st.download_button(
            label=excel_label,
            data=(
                dataframes_to_excel_bytes(
                    excel_sheets
                )
                if excel_has_data
                else b""
            ),
            file_name=f"{safe_file_name}.xlsx",
            mime=EXCEL_MIME_TYPE,
            disabled=not excel_has_data,
            key=f"{key_prefix}_excel_download",
            width="stretch",
        )

    if csv_is_empty and not excel_has_data:
        st.info(no_data_message)


