from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Metric,
    RunReportRequest,
)
from google.oauth2 import service_account

from config.settings import (
    GA4_PROPERTY_ID,
    GA4_SERVICE_ACCOUNT_FILE,
    require_env,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


PAGE_SIZE = 100_000

DIMENSIONS = (
    "date",
    "sessionCampaignName",
    "sessionSourceMedium",
)

METRICS = (
    "sessions",
    "totalUsers",
    "engagedSessions",
    "ecommercePurchases",
    "purchaseRevenue",
)

OUTPUT_COLUMNS = [
    "Date",
    "Campaign",
    "SourceMedium",
    "Sessions",
    "TotalUsers",
    "EngagedSessions",
    "Purchases",
    "PurchaseRevenue",
    "GA4ConversionRate",
    "GA4RevenuePerSession",
    "EngagementRate",
]


def _empty_campaign_frame() -> pd.DataFrame:
    """Return an empty result with a stable downstream schema."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _validate_date_range(
    date_from: str,
    date_to: str,
) -> tuple[str, str]:
    """Validate and normalize an inclusive ISO date range."""
    try:
        start_date = date.fromisoformat(str(date_from).strip())
        end_date = date.fromisoformat(str(date_to).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "GA4 dates must use YYYY-MM-DD format. "
            f"Received: {date_from!r} to {date_to!r}"
        ) from exc

    if start_date > end_date:
        raise ValueError(
            "GA4 start date cannot be after end date. "
            f"Received: {start_date} to {end_date}"
        )

    return start_date.isoformat(), end_date.isoformat()


def _clean_dimension(
    value: str | None,
    fallback: str,
) -> str:
    """Normalize empty and GA4 '(not set)' dimension values."""
    normalized = str(value or "").strip()

    if not normalized or normalized.lower() == "(not set)":
        return fallback

    return normalized


def _metric_value(
    row,
    index: int,
) -> float:
    """Read a GA4 metric safely as a non-null float."""
    try:
        raw_value = row.metric_values[index].value
    except (AttributeError, IndexError):
        return 0.0

    numeric_value = pd.to_numeric(
        raw_value,
        errors="coerce",
    )

    if pd.isna(numeric_value):
        return 0.0

    return float(numeric_value)


class GA4Extractor:
    def __init__(self) -> None:
        property_id = require_env(
            GA4_PROPERTY_ID,
            "GA4_PROPERTY_ID",
        )
        service_account_file = require_env(
            GA4_SERVICE_ACCOUNT_FILE,
            "GA4_SERVICE_ACCOUNT_FILE",
        )

        credentials_path = Path(
            service_account_file
        ).expanduser()

        if not credentials_path.exists():
            raise FileNotFoundError(
                "GA4 service account file was not found: "
                f"{credentials_path}"
            )

        self.credentials = (
            service_account.Credentials
            .from_service_account_file(
                str(credentials_path)
            )
        )

        self.client = BetaAnalyticsDataClient(
            credentials=self.credentials
        )

        self.property_name = (
            f"properties/{property_id.strip()}"
        )

    def _build_request(
        self,
        date_from: str,
        date_to: str,
        offset: int,
    ) -> RunReportRequest:
        return RunReportRequest(
            property=self.property_name,
            dimensions=[
                Dimension(name=name)
                for name in DIMENSIONS
            ],
            metrics=[
                Metric(name=name)
                for name in METRICS
            ],
            date_ranges=[
                DateRange(
                    start_date=date_from,
                    end_date=date_to,
                )
            ],
            limit=PAGE_SIZE,
            offset=offset,
            keep_empty_rows=True,
        )

    def fetch_campaign_performance(
        self,
        date_from: str,
        date_to: str,
    ) -> pd.DataFrame:
        """
        Fetch all GA4 campaign rows for an inclusive date range.

        The method follows GA4 pagination until every available row is
        collected. Ratios are calculated after duplicate grain rows are
        consolidated so downstream reports receive consistent values.
        """
        normalized_from, normalized_to = (
            _validate_date_range(
                date_from,
                date_to,
            )
        )

        rows: list[dict[str, object]] = []
        offset = 0
        reported_row_count: int | None = None

        while True:
            request = self._build_request(
                date_from=normalized_from,
                date_to=normalized_to,
                offset=offset,
            )

            try:
                response = self.client.run_report(
                    request
                )
            except Exception as exc:
                logger.error(
                    "GA4 Data API error for %s to %s "
                    "at offset %s: %s",
                    normalized_from,
                    normalized_to,
                    offset,
                    exc,
                )
                raise

            page_rows = list(response.rows)

            if reported_row_count is None:
                raw_row_count = getattr(
                    response,
                    "row_count",
                    None,
                )

                if raw_row_count is not None:
                    reported_row_count = int(
                        raw_row_count or 0
                    )

            for row in page_rows:
                dimension_values = [
                    value.value
                    for value in row.dimension_values
                ]

                rows.append(
                    {
                        "Date": (
                            dimension_values[0]
                            if len(dimension_values) > 0
                            else ""
                        ),
                        "Campaign": _clean_dimension(
                            (
                                dimension_values[1]
                                if len(dimension_values) > 1
                                else None
                            ),
                            "UNKNOWN",
                        ),
                        "SourceMedium": _clean_dimension(
                            (
                                dimension_values[2]
                                if len(dimension_values) > 2
                                else None
                            ),
                            "UNKNOWN",
                        ),
                        "Sessions": _metric_value(row, 0),
                        "TotalUsers": _metric_value(row, 1),
                        "EngagedSessions": _metric_value(
                            row,
                            2,
                        ),
                        "Purchases": _metric_value(row, 3),
                        "PurchaseRevenue": _metric_value(
                            row,
                            4,
                        ),
                    }
                )

            received_count = len(page_rows)
            offset += received_count

            if received_count == 0:
                break

            if (
                reported_row_count is not None
                and offset >= reported_row_count
            ):
                break

            if received_count < PAGE_SIZE:
                break

        if not rows:
            logger.warning(
                "GA4 returned no campaign data for %s to %s.",
                normalized_from,
                normalized_to,
            )
            return _empty_campaign_frame()

        dataframe = pd.DataFrame(rows)

        dataframe["Date"] = pd.to_datetime(
            dataframe["Date"],
            format="%Y%m%d",
            errors="coerce",
        )

        invalid_date_count = int(
            dataframe["Date"].isna().sum()
        )

        if invalid_date_count:
            logger.warning(
                "Discarding %s GA4 rows with invalid dates.",
                invalid_date_count,
            )

            dataframe = dataframe.loc[
                dataframe["Date"].notna()
            ].copy()

        if dataframe.empty:
            return _empty_campaign_frame()

        numeric_columns = [
            "Sessions",
            "TotalUsers",
            "EngagedSessions",
            "Purchases",
            "PurchaseRevenue",
        ]

        for column in numeric_columns:
            dataframe[column] = (
                pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                )
                .fillna(0.0)
                .astype("float64")
            )

        # Oranları hesaplamadan önce aynı seviyedeki
        # yinelenen GA4 satırlarını birleştir.
        dataframe = (
            dataframe.groupby(
                [
                    "Date",
                    "Campaign",
                    "SourceMedium",
                ],
                as_index=False,
                dropna=False,
            )[numeric_columns]
            .sum()
            .sort_values(
                [
                    "Date",
                    "Campaign",
                    "SourceMedium",
                ]
            )
            .reset_index(drop=True)
        )

        sessions = dataframe["Sessions"]

        positive_sessions = sessions.where(
            sessions > 0
        )

        ratio_sources = {
            "GA4ConversionRate": "Purchases",
            "GA4RevenuePerSession": "PurchaseRevenue",
            "EngagementRate": "EngagedSessions",
        }

        for target_column, source_column in (
            ratio_sources.items()
        ):
            dataframe[target_column] = (
                dataframe[source_column]
                .div(positive_sessions)
                .fillna(0.0)
                .astype("float64")
            )

        logger.info(
            "GA4 campaign data fetched: %s to %s | "
            "API rows: %s | Output rows: %s | Days: %s",
            normalized_from,
            normalized_to,
            len(rows),
            len(dataframe),
            dataframe["Date"].dt.date.nunique(),
        )

        return dataframe[OUTPUT_COLUMNS]